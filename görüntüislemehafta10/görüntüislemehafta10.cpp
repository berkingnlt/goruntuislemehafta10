#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdlib>

using namespace cv;
using namespace std;

struct MaliyetHacmi
{
    int genislik, yukseklik, disparitySayisi;
    int baslangicX, baslangicY;
    vector<int> degerler;

    size_t indeks(int x, int y, int d) const
    {
        return (static_cast<size_t>(y) * genislik + x)
            * disparitySayisi + d;
    }
};

MaliyetHacmi maliyetOlustur(const Mat& sol, const Mat& sag,
    int blokBoyutu, int disparitySayisi)
{
    CV_Assert(sol.type() == CV_8UC1);
    CV_Assert(sag.type() == CV_8UC1);
    CV_Assert(sol.size() == sag.size());
    CV_Assert(blokBoyutu >= 3 && blokBoyutu % 2 == 1);
    CV_Assert(disparitySayisi > 0);

    int yari = blokBoyutu / 2;

    MaliyetHacmi hacim;
    hacim.baslangicX = disparitySayisi - 1 + yari;
    hacim.baslangicY = yari;
    hacim.genislik = sol.cols - hacim.baslangicX - yari;
    hacim.yukseklik = sol.rows - 2 * yari;
    hacim.disparitySayisi = disparitySayisi;

    CV_Assert(hacim.genislik > 0 && hacim.yukseklik > 0);

    hacim.degerler.resize(
        static_cast<size_t>(hacim.genislik)
        * hacim.yukseklik * disparitySayisi
    );

    for (int y = 0; y < hacim.yukseklik; y++)
    {
        int resimY = y + hacim.baslangicY;

        for (int x = 0; x < hacim.genislik; x++)
        {
            int resimX = x + hacim.baslangicX;

            for (int d = 0; d < disparitySayisi; d++)
            {
                int sad = 0;

                for (int dy = -yari; dy <= yari; dy++)
                {
                    const uchar* a = sol.ptr<uchar>(resimY + dy);
                    const uchar* b = sag.ptr<uchar>(resimY + dy);

                    for (int dx = -yari; dx <= yari; dx++)
                    {
                        sad += std::abs(
                            int(a[resimX + dx])
                            - int(b[resimX - d + dx])
                        );
                    }
                }

                hacim.degerler[hacim.indeks(x, y, d)] = sad;
            }
        }
    }

    return hacim;
}

Mat manuelSGM(const MaliyetHacmi& hacim, Size resimBoyutu,
    int P1, int P2)
{
    CV_Assert(P1 >= 0 && P2 >= P1);

    int W = hacim.genislik;
    int H = hacim.yukseklik;
    int D = hacim.disparitySayisi;

    vector<int> toplam(hacim.degerler.size(), 0);

    const int yonler[8][2] = {
        { 1, 0 }, { -1, 0 },
        { 0, 1 }, { 0, -1 },
        { 1, 1 }, { -1, -1 },
        { 1, -1 }, { -1, 1 }
    };

    const int sonsuz = 100000000;

    for (int yon = 0; yon < 8; yon++)
    {
        int dx = yonler[yon][0];
        int dy = yonler[yon][1];

        for (int sy = 0; sy < H; sy++)
        {
            for (int sx = 0; sx < W; sx++)
            {
                int oncekiX = sx - dx;
                int oncekiY = sy - dy;

                if (oncekiX >= 0 && oncekiX < W &&
                    oncekiY >= 0 && oncekiY < H)
                    continue;

                vector<int> onceki(D, 0);
                vector<int> simdiki(D, 0);

                bool ilkPiksel = true;
                int x = sx;
                int y = sy;

                while (x >= 0 && x < W && y >= 0 && y < H)
                {
                    int minimumOnceki = 0;

                    if (!ilkPiksel)
                    {
                        minimumOnceki = *min_element(
                            onceki.begin(), onceki.end()
                        );
                    }

                    for (int d = 0; d < D; d++)
                    {
                        size_t indis = hacim.indeks(x, y, d);
                        int maliyet = hacim.degerler[indis];

                        if (ilkPiksel)
                        {
                            simdiki[d] = maliyet;
                        }
                        else
                        {
                            int ayni = onceki[d];

                            int birAz = (d > 0)
                                ? onceki[d - 1] + P1 : sonsuz;

                            int birFazla = (d + 1 < D)
                                ? onceki[d + 1] + P1 : sonsuz;

                            int buyukDegisim = minimumOnceki + P2;

                            int enIyi = std::min(
                                std::min(ayni, birAz),
                                std::min(birFazla, buyukDegisim)
                            );

                            simdiki[d] =
                                maliyet + enIyi - minimumOnceki;
                        }

                        toplam[indis] += simdiki[d];
                    }

                    onceki.swap(simdiki);
                    ilkPiksel = false;

                    x += dx;
                    y += dy;
                }
            }
        }
    }

    Mat disparity(resimBoyutu, CV_32F, Scalar(-1));

    for (int y = 0; y < H; y++)
    {
        for (int x = 0; x < W; x++)
        {
            int enIyiD = 0;
            int enIyiMaliyet = sonsuz;

            for (int d = 0; d < D; d++)
            {
                int maliyet = toplam[hacim.indeks(x, y, d)];

                if (maliyet < enIyiMaliyet)
                {
                    enIyiMaliyet = maliyet;
                    enIyiD = d;
                }
            }

            disparity.at<float>(
                y + hacim.baslangicY,
                x + hacim.baslangicX
            ) = static_cast<float>(enIyiD);
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

        const int blokBoyutu = 5;
        const int disparitySayisi = 32;

        TickMeter maliyetSuresi;
        maliyetSuresi.start();

        MaliyetHacmi hacim = maliyetOlustur(
            solGri, sagGri, blokBoyutu, disparitySayisi
        );

        maliyetSuresi.stop();

        int p1Degerleri[] = { 0, 200, 800 };
        int p2Degerleri[] = { 0, 800, 3200 };

        string adlar[] = {
            "Cezasiz", "Orta", "Yuksek"
        };

        ofstream rapor("4_olcumler.txt");
        FileStorage veriler(
            "4_disparity_verileri.yml", FileStorage::WRITE
        );

        if (!rapor.is_open() || !veriler.isOpened())
        {
            cerr << "Sonuc dosyalari acilamadi.\n";
            return 1;
        }

        rapor << "Blok boyutu: " << blokBoyutu << "\n";
        rapor << "Disparity araligi: 0-31\n";
        rapor << "Yon sayisi: 8\n";
        rapor << "Ortak maliyet hacmi suresi: "
            << maliyetSuresi.getTimeMilli() << " ms\n\n";

        Mat renkliSonuclar[3];

        for (int i = 0; i < 3; i++)
        {
            cout << adlar[i] << " hesaplaniyor...\n";

            TickMeter sure;
            sure.start();

            Mat disparity = manuelSGM(
                hacim, sol.size(),
                p1Degerleri[i], p2Degerleri[i]
            );

            sure.stop();

            Mat griHarita;
            Mat pozitifMaske = disparity > 0;
            Mat hesaplananMaske = disparity >= 0;

            disparity.convertTo(
                griHarita, CV_8U,
                255.0 / (disparitySayisi - 1)
            );

            griHarita.setTo(0, pozitifMaske == 0);

            applyColorMap(
                griHarita, renkliSonuclar[i], COLORMAP_JET
            );

            renkliSonuclar[i].setTo(
                Scalar(0, 0, 0), pozitifMaske == 0
            );

            string onEk = "4_" + adlar[i];

            bool kaydedildi = true;
            kaydedildi &= imwrite(
                onEk + "_gri.png", griHarita
            );
            kaydedildi &= imwrite(
                onEk + "_renkli.png", renkliSonuclar[i]
            );
            kaydedildi &= imwrite(
                onEk + "_maske.png", hesaplananMaske
            );

            if (!kaydedildi)
            {
                cerr << "PNG dosyalari kaydedilemedi.\n";
                return 1;
            }

            veriler << (adlar[i] + "_disparity") << disparity;
            veriler << (adlar[i] + "_maske") << hesaplananMaske;

            rapor << adlar[i]
                << " P1=" << p1Degerleri[i]
                << " P2=" << p2Degerleri[i] << "\n";

            rapor << "Birikim ve disparity secimi: "
                << sure.getTimeMilli() << " ms\n";

            rapor << "Maliyet olusturma dahil toplam: "
                << maliyetSuresi.getTimeMilli()
                + sure.getTimeMilli()
                << " ms\n\n";

            cout << adlar[i] << ": "
                << maliyetSuresi.getTimeMilli()
                + sure.getTimeMilli()
                << " ms\n";
        }

        Mat ilkIki, panel;
        hconcat(renkliSonuclar[0], renkliSonuclar[1], ilkIki);
        hconcat(ilkIki, renkliSonuclar[2], panel);

        if (!imwrite("4_Smoothness_karsilastirma.png", panel))
        {
            cerr << "Karsilastirma resmi kaydedilemedi.\n";
            return 1;
        }

        veriler.release();
        rapor.close();

        cout << "\nSoldan saga: Cezasiz - Orta - Yuksek\n";
        cout << "Sonuclar kaydedildi.\n";

        namedWindow("Manuel SGM - Smoothness", WINDOW_NORMAL);
        imshow("Manuel SGM - Smoothness", panel);

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