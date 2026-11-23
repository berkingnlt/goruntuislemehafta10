#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <limits>
#include <cmath>

using namespace cv;
using namespace std;

using BenzerlikFonksiyonu = double (*)(
    const Mat&, const Mat&, int, int, int, int
    );

double SAD(const Mat& sol, const Mat& sag,
    int x, int y, int sagX, int yari)
{
    double toplam = 0;

    for (int dy = -yari; dy <= yari; dy++)
    {
        const uchar* a = sol.ptr<uchar>(y + dy);
        const uchar* b = sag.ptr<uchar>(y + dy);

        for (int dx = -yari; dx <= yari; dx++)
        {
            double fark = double(a[x + dx]) - b[sagX + dx];
            toplam += std::abs(fark);
        }
    }

    return toplam;
}

double SSD(const Mat& sol, const Mat& sag,
    int x, int y, int sagX, int yari)
{
    double toplam = 0;

    for (int dy = -yari; dy <= yari; dy++)
    {
        const uchar* a = sol.ptr<uchar>(y + dy);
        const uchar* b = sag.ptr<uchar>(y + dy);

        for (int dx = -yari; dx <= yari; dx++)
        {
            double fark = double(a[x + dx]) - b[sagX + dx];
            toplam += fark * fark;
        }
    }

    return toplam;
}

double ZNCC(const Mat& sol, const Mat& sag,
    int x, int y, int sagX, int yari)
{
    double toplamA = 0, toplamB = 0;
    double toplamAA = 0, toplamBB = 0, toplamAB = 0;

    int kenar = 2 * yari + 1;
    double n = double(kenar) * kenar;

    for (int dy = -yari; dy <= yari; dy++)
    {
        const uchar* a = sol.ptr<uchar>(y + dy);
        const uchar* b = sag.ptr<uchar>(y + dy);

        for (int dx = -yari; dx <= yari; dx++)
        {
            double p = a[x + dx];
            double q = b[sagX + dx];

            toplamA += p;
            toplamB += q;
            toplamAA += p * p;
            toplamBB += q * q;
            toplamAB += p * q;
        }
    }

    double varyansA = n * toplamAA - toplamA * toplamA;
    double varyansB = n * toplamBB - toplamB * toplamB;

    if (varyansA <= 1e-9 || varyansB <= 1e-9)
        return numeric_limits<double>::infinity();

    double pay = n * toplamAB - toplamA * toplamB;
    double korelasyon = pay / sqrt(varyansA * varyansB);

    if (korelasyon > 1.0) korelasyon = 1.0;
    if (korelasyon < -1.0) korelasyon = -1.0;

    return 1.0 - korelasyon;
}

Mat blokEslestirme(const Mat& sol, const Mat& sag,
    int blokBoyutu, int disparitySayisi,
    BenzerlikFonksiyonu maliyetFonksiyonu)
{
    CV_Assert(sol.type() == CV_8UC1);
    CV_Assert(sag.type() == CV_8UC1);
    CV_Assert(sol.size() == sag.size());
    CV_Assert(blokBoyutu >= 3 && blokBoyutu % 2 == 1);
    CV_Assert(disparitySayisi > 0);
    CV_Assert(maliyetFonksiyonu != nullptr);

    Mat disparity(sol.size(), CV_32F, Scalar(-1));

    int yari = blokBoyutu / 2;
    int maksimumDisparity = disparitySayisi - 1;

    for (int y = yari; y < sol.rows - yari; y++)
    {
        for (int x = maksimumDisparity + yari;
            x < sol.cols - yari; x++)
        {
            double enIyiMaliyet =
                numeric_limits<double>::infinity();

            int enIyiDisparity = -1;

            for (int d = 0; d < disparitySayisi; d++)
            {
                double maliyet = maliyetFonksiyonu(
                    sol, sag, x, y, x - d, yari
                );

                if (maliyet < enIyiMaliyet)
                {
                    enIyiMaliyet = maliyet;
                    enIyiDisparity = d;
                }
            }

            disparity.at<float>(y, x) =
                static_cast<float>(enIyiDisparity);
        }
    }

    return disparity;
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
            cerr << "Goruntu boyutlari ayni olmali.\n";
            return 1;
        }

        Mat solGri, sagGri;
        cvtColor(sol, solGri, COLOR_BGR2GRAY);
        cvtColor(sag, sagGri, COLOR_BGR2GRAY);

        const int blokBoyutu = 7;
        const int disparitySayisi = 32;

        string adlar[] = { "SAD", "SSD", "ZNCC" };

        BenzerlikFonksiyonu fonksiyonlar[] = {
            SAD, SSD, ZNCC
        };

        ofstream rapor("3_olcumler.txt");
        FileStorage veriler(
            "3_disparity_verileri.yml", FileStorage::WRITE
        );

        if (!rapor.is_open() || !veriler.isOpened())
        {
            cerr << "Sonuc dosyalari acilamadi.\n";
            return 1;
        }

        rapor << "Blok boyutu: " << blokBoyutu << "\n";
        rapor << "Disparity araligi: 0-"
            << disparitySayisi - 1 << "\n\n";

        Mat renkliSonuclar[3];

        for (int i = 0; i < 3; i++)
        {
            cout << adlar[i] << " hesaplaniyor...\n";

            TickMeter sure;
            sure.start();

            Mat disparity = blokEslestirme(
                solGri, sagGri,
                blokBoyutu, disparitySayisi,
                fonksiyonlar[i]
            );

            sure.stop();

            Mat hesaplanan = disparity >= 0;
            Mat pozitif = disparity > 0;
            Mat griHarita;

            disparity.convertTo(
                griHarita, CV_8U,
                255.0 / (disparitySayisi - 1)
            );

            griHarita.setTo(0, pozitif == 0);

            applyColorMap(
                griHarita, renkliSonuclar[i], COLORMAP_JET
            );

            renkliSonuclar[i].setTo(
                Scalar(0, 0, 0), pozitif == 0
            );

            string onEk = "3_" + adlar[i];

            bool kaydedildi = true;
            kaydedildi &= imwrite(onEk + "_gri.png", griHarita);
            kaydedildi &= imwrite(
                onEk + "_renkli.png", renkliSonuclar[i]
            );
            kaydedildi &= imwrite(
                onEk + "_hesaplanan_maske.png", hesaplanan
            );

            if (!kaydedildi)
            {
                cerr << "PNG dosyalari kaydedilemedi.\n";
                return 1;
            }

            veriler << (adlar[i] + "_disparity") << disparity;
            veriler << (adlar[i] + "_maske") << hesaplanan;

            double oran =
                100.0 * countNonZero(hesaplanan) / disparity.total();

            cout << adlar[i] << ": "
                << sure.getTimeMilli() << " ms\n";

            rapor << adlar[i] << "\n";
            rapor << "Hesaplama suresi: "
                << sure.getTimeMilli() << " ms\n";
            rapor << "Disparity atanan piksel orani: %"
                << oran << "\n\n";
        }

        Mat ilkIki, panel;
        hconcat(renkliSonuclar[0], renkliSonuclar[1], ilkIki);
        hconcat(ilkIki, renkliSonuclar[2], panel);

        if (!imwrite("3_SAD_SSD_ZNCC.png", panel))
        {
            cerr << "Karsilastirma resmi kaydedilemedi.\n";
            return 1;
        }

        veriler.release();
        rapor.close();

        cout << "\nSoldan saga: SAD - SSD - ZNCC\n";
        cout << "Sonuclar kaydedildi.\n";

        namedWindow("SAD - SSD - ZNCC", WINDOW_NORMAL);
        imshow("SAD - SSD - ZNCC", panel);

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