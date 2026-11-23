#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <limits>
#include <cstdlib>
#include <cmath>

using namespace cv;
using namespace std;

Mat blokEslestir(const Mat& a, const Mat& b,
    const Mat& kenarA, const Mat& kenarB,
    bool seyrek, bool ters,
    int blokBoyutu, int disparitySayisi)
{
    Mat disparity(a.size(), CV_32F, Scalar(-1));

    int yari = blokBoyutu / 2;
    int maksimumD = disparitySayisi - 1;

    int baslangicX = ters ? yari : maksimumD + yari;
    int bitisX = ters
        ? a.cols - maksimumD - yari
        : a.cols - yari;

    for (int y = yari; y < a.rows - yari; y++)
    {
        for (int x = baslangicX; x < bitisX; x++)
        {
            if (seyrek && kenarA.at<uchar>(y, x) == 0)
                continue;

            int enIyiMaliyet = numeric_limits<int>::max();
            int enIyiD = -1;

            for (int d = 0; d < disparitySayisi; d++)
            {
                int adayX = ters ? x + d : x - d;

                if (seyrek && kenarB.at<uchar>(y, adayX) == 0)
                    continue;

                int sad = 0;

                for (int dy = -yari; dy <= yari; dy++)
                {
                    const uchar* satirA = a.ptr<uchar>(y + dy);
                    const uchar* satirB = b.ptr<uchar>(y + dy);

                    for (int dx = -yari; dx <= yari; dx++)
                    {
                        sad += std::abs(
                            int(satirA[x + dx])
                            - int(satirB[adayX + dx])
                        );
                    }
                }

                if (sad < enIyiMaliyet)
                {
                    enIyiMaliyet = sad;
                    enIyiD = d;
                }
            }

            disparity.at<float>(y, x) =
                static_cast<float>(enIyiD);
        }
    }

    return disparity;
}

Mat tutarlilikKontrolu(const Mat& solDisparity,
    const Mat& sagDisparity)
{
    Mat sonuc(solDisparity.size(), CV_32F, Scalar(-1));

    for (int y = 0; y < solDisparity.rows; y++)
    {
        for (int x = 0; x < solDisparity.cols; x++)
        {
            float dSol = solDisparity.at<float>(y, x);

            if (dSol <= 0)
                continue;

            int sagX = x - cvRound(dSol);

            if (sagX < 0 || sagX >= sagDisparity.cols)
                continue;

            float dSag = sagDisparity.at<float>(y, sagX);

            if (dSag > 0 && std::abs(dSol - dSag) <= 1.0f)
                sonuc.at<float>(y, x) = dSol;
        }
    }

    return sonuc;
}

void haritaKaydet(const string& ad, const Mat& disparity,
    int disparitySayisi, Mat& renkli)
{
    Mat maske = disparity > 0;
    Mat gri;

    disparity.convertTo(
        gri, CV_8U, 255.0 / (disparitySayisi - 1)
    );

    gri.setTo(0, maske == 0);

    applyColorMap(gri, renkli, COLORMAP_JET);
    renkli.setTo(Scalar(0, 0, 0), maske == 0);

    if (!imwrite(ad + "_gri.png", gri) ||
        !imwrite(ad + "_renkli.png", renkli) ||
        !imwrite(ad + "_maske.png", maske))
    {
        CV_Error(Error::StsError, "Resimler kaydedilemedi.");
    }
}

void olcumYaz(ostream& cikti, const string& ad,
    const Mat& ham, const Mat& sonuc,
    const Mat& sol, const Mat& sag,
    const Mat& ortakAlan, double sure)
{
    Mat adayMaske = (ham > 0) & ortakAlan;
    Mat gecerliMaske = (sonuc > 0) & ortakAlan;

    int alan = countNonZero(ortakAlan);
    int aday = countNonZero(adayMaske);
    int gecerli = countNonZero(gecerliMaske);

    double toplamFark = 0;

    for (int y = 0; y < sol.rows; y++)
    {
        for (int x = 0; x < sol.cols; x++)
        {
            if (gecerliMaske.at<uchar>(y, x) == 0)
                continue;

            int d = cvRound(sonuc.at<float>(y, x));

            toplamFark += std::abs(
                int(sol.at<uchar>(y, x))
                - int(sag.at<uchar>(y, x - d))
            );
        }
    }

    cikti << ad << "\n";
    cikti << "Toplam sure: " << sure << " ms\n";
    cikti << "Pozitif eslesme adayi: " << aday << "\n";
    cikti << "Tutarlilik kontrolunu gecen: " << gecerli << "\n";

    cikti << "Ortak alanda kapsama: %"
        << 100.0 * gecerli / alan << "\n";

    if (aday > 0)
    {
        cikti << "Sol-sag kontrolunu gecme orani: %"
            << 100.0 * gecerli / aday << "\n";
    }
    else
    {
        cikti << "Sol-sag kontrolunu gecme orani: hesaplanamadi\n";
    }

    if (gecerli > 0)
    {
        cikti << "Eslesen piksellerin fotometrik MAE degeri: "
            << toplamFark / gecerli << "\n\n";
    }
    else
    {
        cikti << "Fotometrik MAE: hesaplanamadi\n\n";
    }
}

int main()
{
    try
    {
        Mat sol = imread("scene1.row3.col1.ppm", IMREAD_COLOR);
        Mat sag = imread("scene1.row3.col3.ppm", IMREAD_COLOR);

        if (sol.empty() || sag.empty())
        {
            cerr << "PPM dosyalari bulunamadi.\n";
            return 1;
        }

        if (sol.size() != sag.size())
        {
            cerr << "Goruntulerin boyutlari ayni olmali.\n";
            return 1;
        }

        Mat solGri, sagGri;
        cvtColor(sol, solGri, COLOR_BGR2GRAY);
        cvtColor(sag, sagGri, COLOR_BGR2GRAY);

        const int blokBoyutu = 7;
        const int disparitySayisi = 32;
        const int yari = blokBoyutu / 2;
        const int kenarPayi = disparitySayisi - 1 + yari;

        if (sol.cols <= 2 * kenarPayi ||
            sol.rows <= 2 * yari)
        {
            cerr << "Goruntuler bu parametreler icin cok kucuk.\n";
            return 1;
        }

        cout << "Dense eslestirme hesaplaniyor...\n";

        TickMeter denseSure;
        denseSure.start();

        Mat denseSol = blokEslestir(
            solGri, sagGri, Mat(), Mat(),
            false, false, blokBoyutu, disparitySayisi
        );

        Mat denseSag = blokEslestir(
            sagGri, solGri, Mat(), Mat(),
            false, true, blokBoyutu, disparitySayisi
        );

        Mat denseSonuc = tutarlilikKontrolu(denseSol, denseSag);

        denseSure.stop();

        cout << "Sparse eslestirme hesaplaniyor...\n";

        TickMeter sparseSure;
        sparseSure.start();

        Mat solYumusak, sagYumusak;
        Mat solKenar, sagKenar;

        GaussianBlur(solGri, solYumusak, Size(3, 3), 0.8);
        GaussianBlur(sagGri, sagYumusak, Size(3, 3), 0.8);

        Canny(solYumusak, solKenar, 50, 150);
        Canny(sagYumusak, sagKenar, 50, 150);

        Mat sparseSol = blokEslestir(
            solGri, sagGri, solKenar, sagKenar,
            true, false, blokBoyutu, disparitySayisi
        );

        Mat sparseSag = blokEslestir(
            sagGri, solGri, sagKenar, solKenar,
            true, true, blokBoyutu, disparitySayisi
        );

        Mat sparseSonuc = tutarlilikKontrolu(sparseSol, sparseSag);

        sparseSure.stop();

        Mat ortakAlan(sol.size(), CV_8U, Scalar(0));

        ortakAlan(Rect(
            kenarPayi,
            yari,
            sol.cols - 2 * kenarPayi,
            sol.rows - 2 * yari
        )).setTo(255);

        Mat denseRenkli, sparseRenkli;

        haritaKaydet(
            "Odev_dense", denseSonuc,
            disparitySayisi, denseRenkli
        );

        haritaKaydet(
            "Odev_sparse", sparseSonuc,
            disparitySayisi, sparseRenkli
        );

        if (!imwrite("Odev_sol_kenarlar.png", solKenar) ||
            !imwrite("Odev_sag_kenarlar.png", sagKenar) ||
            !imwrite("Odev_ortak_alan.png", ortakAlan))
        {
            cerr << "Kenar resimleri kaydedilemedi.\n";
            return 1;
        }

        Mat panel;
        hconcat(denseRenkli, sparseRenkli, panel);

        if (!imwrite("Odev_dense_sparse.png", panel))
        {
            cerr << "Karsilastirma resmi kaydedilemedi.\n";
            return 1;
        }

        FileStorage veri(
            "Odev_disparity_verileri.yml", FileStorage::WRITE
        );
        ofstream rapor("Odev_olcumler.txt");

        if (!veri.isOpened() || !rapor.is_open())
        {
            cerr << "Veri dosyalari acilamadi.\n";
            return 1;
        }

        veri << "dense_ham" << denseSol;
        veri << "dense_filtreli" << denseSonuc;
        veri << "sparse_ham" << sparseSol;
        veri << "sparse_filtreli" << sparseSonuc;
        veri << "ortak_alan" << ortakAlan;
        veri.release();

        olcumYaz(
            cout, "DENSE", denseSol, denseSonuc,
            solGri, sagGri, ortakAlan, denseSure.getTimeMilli()
        );

        olcumYaz(
            cout, "SPARSE", sparseSol, sparseSonuc,
            solGri, sagGri, ortakAlan, sparseSure.getTimeMilli()
        );

        olcumYaz(
            rapor, "DENSE", denseSol, denseSonuc,
            solGri, sagGri, ortakAlan, denseSure.getTimeMilli()
        );

        olcumYaz(
            rapor, "SPARSE", sparseSol, sparseSonuc,
            solGri, sagGri, ortakAlan, sparseSure.getTimeMilli()
        );

        double hizOrani = denseSure.getTimeMilli()
            / sparseSure.getTimeMilli();

        cout << "Dense suresi / Sparse suresi: "
            << hizOrani << "\n";

        rapor << "Dense suresi / Sparse suresi: "
            << hizOrani << "\n";
        rapor << "Sparse suresine Gaussian ve Canny dahildir.\n";
        rapor << "Okuma, kaydetme ve gorsellestirme harictir.\n";
        rapor << "Fotometrik MAE disparity dogrulugu degildir.\n";

        rapor.close();

        namedWindow("Sol kenarlar", WINDOW_NORMAL);
        imshow("Sol kenarlar", solKenar);

        namedWindow("Sag kenarlar", WINDOW_NORMAL);
        imshow("Sag kenarlar", sagKenar);

        namedWindow("Dense - Sparse", WINDOW_NORMAL);
        imshow("Dense - Sparse", panel);

        cout << "\nSolda Dense, sagda Sparse.\n";
        cout << "Sonuclar kaydedildi.\n";

        waitKey(0);
        destroyAllWindows();
    }
    catch (const cv::Exception& hata)
    {
        cerr << hata.what() << "\n";
        return 1;
    }

    return 0;
}