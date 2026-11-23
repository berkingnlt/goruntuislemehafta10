#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

void haritaHazirla(const Mat& ham, int disparitySayisi,
    Mat& disparity, Mat& maske,
    Mat& gri, Mat& renkli)
{
    ham.convertTo(disparity, CV_32F, 1.0 / 16.0);

    maske = disparity > 0;

    disparity.convertTo(
        gri, CV_8U, 255.0 / (disparitySayisi - 1)
    );

    gri.setTo(0, maske == 0);

    applyColorMap(gri, renkli, COLORMAP_JET);
    renkli.setTo(Scalar(0, 0, 0), maske == 0);
}

void goster(const string& ad, const Mat& resim)
{
    namedWindow(ad, WINDOW_NORMAL);
    imshow(ad, resim);
}

int main()
{
    try
    {
        Mat sol = imread("scene1.row3.col1.ppm", IMREAD_COLOR);
        Mat sag = imread("scene1.row3.col3.ppm", IMREAD_COLOR);

        if (sol.empty() || sag.empty())
        {
            cout << "PPM dosyalari okunamadi.\n";
            cout << "Dosyalari proje klasorune koyunuz.\n";
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

        const int disparitySayisi = 32;

        Ptr<StereoBM> bm = StereoBM::create(disparitySayisi, 15);
        bm->setMinDisparity(0);
        bm->setTextureThreshold(10);
        bm->setUniquenessRatio(10);
        bm->setSpeckleWindowSize(50);
        bm->setSpeckleRange(2);
        bm->setDisp12MaxDiff(1);

        const int blokBoyutu = 5;

        Ptr<StereoSGBM> sgbm = StereoSGBM::create(
            0,
            disparitySayisi,
            blokBoyutu
        );

        sgbm->setP1(8 * blokBoyutu * blokBoyutu);
        sgbm->setP2(32 * blokBoyutu * blokBoyutu);
        sgbm->setPreFilterCap(31);
        sgbm->setUniquenessRatio(10);
        sgbm->setSpeckleWindowSize(50);
        sgbm->setSpeckleRange(2);
        sgbm->setDisp12MaxDiff(1);
        sgbm->setMode(StereoSGBM::MODE_SGBM);

        Mat bmHam, sgbmHam;

        TickMeter bmSure, sgbmSure;

        bmSure.start();
        bm->compute(solGri, sagGri, bmHam);
        bmSure.stop();

        sgbmSure.start();
        sgbm->compute(solGri, sagGri, sgbmHam);
        sgbmSure.stop();

        Mat bmDisparity, bmMaske, bmGri, bmRenkli;
        Mat sgbmDisparity, sgbmMaske, sgbmGri, sgbmRenkli;

        haritaHazirla(bmHam, disparitySayisi,
            bmDisparity, bmMaske, bmGri, bmRenkli);

        haritaHazirla(sgbmHam, disparitySayisi,
            sgbmDisparity, sgbmMaske, sgbmGri, sgbmRenkli);

        imwrite("1_sol.png", sol);
        imwrite("1_sag.png", sag);
        imwrite("1_BM_gri.png", bmGri);
        imwrite("1_BM_renkli.png", bmRenkli);
        imwrite("1_SGBM_gri.png", sgbmGri);
        imwrite("1_SGBM_renkli.png", sgbmRenkli);

        FileStorage dosya("1_disparity_verileri.yml", FileStorage::WRITE);

        if (!dosya.isOpened())
        {
            cerr << "Sayisal sonuc dosyasi acilamadi.\n";
            return 1;
        }

        dosya << "BM_disparity_piksel" << bmDisparity;
        dosya << "BM_gecerli_maske" << bmMaske;
        dosya << "SGBM_disparity_piksel" << sgbmDisparity;
        dosya << "SGBM_gecerli_maske" << sgbmMaske;
        dosya.release();

        Mat bmPanel, sgbmPanel;

        hconcat(sol, bmRenkli, bmPanel);
        hconcat(sol, sgbmRenkli, sgbmPanel);

        imwrite("1_BM_karsilastirma.png", bmPanel);
        imwrite("1_SGBM_karsilastirma.png", sgbmPanel);

        cout << "Goruntu boyutu: "
            << sol.cols << " x " << sol.rows << "\n";

        cout << "StereoBM hesaplama suresi: "
            << bmSure.getTimeMilli() << " ms\n";

        cout << "StereoSGBM hesaplama suresi: "
            << sgbmSure.getTimeMilli() << " ms\n";

        cout << "BM pozitif disparity orani: %"
            << 100.0 * countNonZero(bmMaske) / bmMaske.total()
            << "\n";

        cout << "SGBM pozitif disparity orani: %"
            << 100.0 * countNonZero(sgbmMaske) / sgbmMaske.total()
            << "\n";

        cout << "\nSonuclar kaydedildi.\n";
        cout << "Pencereleri kapatmak icin bir tusa basin.\n";

        goster("Sol goruntu", sol);
        goster("Sag goruntu", sag);
        goster("StereoBM - Sol ve disparity", bmPanel);
        goster("StereoSGBM - Sol ve disparity", sgbmPanel);
        goster("StereoBM - Gri disparity", bmGri);
        goster("StereoSGBM - Gri disparity", sgbmGri);

        waitKey(0);
        destroyAllWindows();
    }
    catch (const cv::Exception& hata)
    {
        cerr << "OpenCV hatasi:\n" << hata.what() << "\n";
        return 1;
    }

    return 0;
}