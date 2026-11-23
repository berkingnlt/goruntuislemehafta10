#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <limits>
#include <cstdlib>

using namespace cv;
using namespace std;

Mat blokEslestirme(const Mat& sol, const Mat& sag,
    int blokBoyutu, int disparitySayisi)
{
    CV_Assert(sol.type() == CV_8UC1);
    CV_Assert(sag.type() == CV_8UC1);
    CV_Assert(sol.size() == sag.size());
    CV_Assert(blokBoyutu >= 3 && blokBoyutu % 2 == 1);
    CV_Assert(disparitySayisi > 0);

    Mat disparity(sol.size(), CV_32F, Scalar(-1));

    int yari = blokBoyutu / 2;
    int maksimumDisparity = disparitySayisi - 1;

    for (int y = yari; y < sol.rows - yari; y++)
    {
        for (int x = maksimumDisparity + yari;
            x < sol.cols - yari; x++)
        {
            int enIyiMaliyet = numeric_limits<int>::max();
            int enIyiDisparity = -1;

            for (int d = 0; d < disparitySayisi; d++)
            {
                int sagX = x - d;
                int sad = 0;

                for (int dy = -yari; dy <= yari; dy++)
                {
                    const uchar* solSatir = sol.ptr<uchar>(y + dy);
                    const uchar* sagSatir = sag.ptr<uchar>(y + dy);

                    for (int dx = -yari; dx <= yari; dx++)
                    {
                        int solDeger = solSatir[x + dx];
                        int sagDeger = sagSatir[sagX + dx];

                        sad += std::abs(solDeger - sagDeger);
                    }
                }

                if (sad < enIyiMaliyet)
                {
                    enIyiMaliyet = sad;
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
            cout << "PPM dosyalari bulunamadi.\n";
            return 1;
        }

        if (sol.size() != sag.size())
        {
            cout << "Goruntulerin boyutlari ayni olmali.\n";
            return 1;
        }

        Mat solGri, sagGri;
        cvtColor(sol, solGri, COLOR_BGR2GRAY);
        cvtColor(sag, sagGri, COLOR_BGR2GRAY);

        const int blokBoyutu = 7;
        const int disparitySayisi = 32;

        TickMeter sure;
        sure.start();

        Mat disparity = blokEslestirme(
            solGri, sagGri, blokBoyutu, disparitySayisi
        );

        sure.stop();

        Mat hesaplananMaske = disparity >= 0;
        Mat pozitifMaske = disparity > 0;

        Mat griHarita;
        disparity.convertTo(
            griHarita, CV_8U, 255.0 / (disparitySayisi - 1)
        );
        griHarita.setTo(0, pozitifMaske == 0);

        Mat renkliHarita;
        applyColorMap(griHarita, renkliHarita, COLORMAP_JET);
        renkliHarita.setTo(Scalar(0, 0, 0), pozitifMaske == 0);

        Mat ilkPanel, panel;
        hconcat(sol, sag, ilkPanel);
        hconcat(ilkPanel, renkliHarita, panel);

        bool kaydedildi = true;
        kaydedildi &= imwrite("2_manuel_gri.png", griHarita);
        kaydedildi &= imwrite("2_manuel_renkli.png", renkliHarita);
        kaydedildi &= imwrite("2_karsilastirma.png", panel);
        kaydedildi &= imwrite("2_hesaplanan_maske.png", hesaplananMaske);

        FileStorage dosya(
            "2_disparity_verileri.yml", FileStorage::WRITE
        );

        if (!dosya.isOpened() || !kaydedildi)
        {
            cerr << "Sonuc dosyalarinin kaydinda hata olustu.\n";
            return 1;
        }

        dosya << "disparity_piksel" << disparity;
        dosya << "hesaplanan_maske" << hesaplananMaske;
        dosya << "blok_boyutu" << blokBoyutu;
        dosya << "disparity_sayisi" << disparitySayisi;
        dosya.release();

        cout << "Yontem: Manuel blok eslestirme - SAD\n";
        cout << "Blok boyutu: " << blokBoyutu
            << " x " << blokBoyutu << "\n";
        cout << "Aranan disparity: 0 - "
            << disparitySayisi - 1 << " piksel\n";
        cout << "Hesaplama suresi: "
            << sure.getTimeMilli() << " ms\n";
        cout << "Sonuclar kaydedildi.\n";

        namedWindow("Sol - Sag - Manuel SAD", WINDOW_NORMAL);
        imshow("Sol - Sag - Manuel SAD", panel);

        namedWindow("Manuel gri disparity", WINDOW_NORMAL);
        imshow("Manuel gri disparity", griHarita);

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