#include <string.h>
#include <vector>
#include <stdio.h>
#include <conio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml.hpp>
#include "dlib/all/source.cpp"
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>

using namespace dlib;
using namespace std;
using namespace cv;
using namespace cv::ml;

wstring emotionLabels = L"C:\\Users\\ִלטענטי\\Desktop\\Emotion_labels\\Emotion";
wstring emotionImgs = L"C:\\Users\\ִלטענטי\\Desktop\\extended-cohn-kanade-images\\cohn-kanade-images";
wstring emotionSort = L"C:\\Users\\ִלטענטי\\Desktop\\extended-cohn-kanade-images\\Emotion";
wstring emotionTest = L"C:\\Users\\ִלטענטי\\Desktop\\extended-cohn-kanade-images\\Test";
wstring emotionNormalized = L"C:\\Users\\ִלטענטי\\Desktop\\extended-cohn-kanade-images\\Normalized";

const int countEmotion = 8;

shape_predictor sp;
frontal_face_detector detector;

struct Landmarks
{
	std::vector<float> dist;
	std::vector<Point2f> rel;
};

void SortAndLandmarkImgs();
void drawLandmark(Mat imgName, Landmarks *lm = new Landmarks(), string outFileName = "", bool writeCSV = false);
void learn();
void parseCSV(wstring fileName);

void getAllFilesInDir(wstring strPath, std::vector<wstring> & strFiles);
std::vector<wstring> parseFileName(wstring fileName);
void copyFile(const std::wstring& fileNameFrom, const std::wstring& fileNameTo);
std::string wstringToString(std::wstring const &s, std::locale const &loc, char default_char = '?');
void splitString(const string &fullstr, std::vector<string> &elements, const string &delimiter);
float stringToFloat(const string &str);
void readData(const wstring &filename, const string &csvdelimiter, std::vector< float > &sarr);

template<class T>
T innerAngle(cv::Point_<T> &p1, cv::Point_<T> &p2, cv::Point_<T> &c);
float lengthLine(cv::Point2f &p1, cv::Point2f &p2);
void readCSVPoint(Mat &trainingData, Mat &labels);
void readCSVDist(Mat &trainingData, Mat &labels);
void readCSVCustomDist(Mat &trainingData, Mat &labels);
void imageProcessing(Mat &cvImg, std::vector<float> &distances, std::vector<Point2f> &relativePoints);
void imagesToVideo();
void markingVideo();

string classEmotion[8] = {"Angry", "Contempt", "Disgust", "Fear", "Happy", "Sorrow", "Surprise", "Neutral"};

void allocationFaceComponent(Mat imgName, string fileName, string outFileName);

int main()
{ 
	detector = get_frontal_face_detector();
	deserialize("C:\\Users\\ִלטענטי\\Desktop\\dlib-18.18\\shape_predictor_68_face_landmarks.dat") >> sp;


	SortAndLandmarkImgs();
	return 0;

	Ptr<cv::ml::ANN_MLP> model = Algorithm::load<cv::ml::ANN_MLP>("ANN_dist_4layers.yml");
	std::vector<float> distances;
	std::vector<Point2f> relativePoints;
	std::vector<std::vector<Point2f>> _points;
	std::vector<float> relP;
	int index = countEmotion - 1;
	float maxResult = -1;
	
	cv::VideoCapture cap("C:\\Users\\ִלטענטי\\Desktop\\Console\\Console\\t2.avi"); // open the default camera
	if (!cap.isOpened())  // check if we succeeded
	{
		cout << "Cap not opening" << endl;
		return -1;
	}

	int width = static_cast<int>(cap.get(CV_CAP_PROP_FRAME_WIDTH));
	int height = static_cast<int>(cap.get(CV_CAP_PROP_FRAME_HEIGHT));
	//cv::VideoWriter videoWriter("example.avi", CV_FOURCC('M', 'P', 'E', 'G'), 10, Size(width, height));
	/*cv::VideoWriter videoWriter("example.avi", CV_FOURCC('D', 'I', 'V', 'X'), 10, Size(width, height));
	if (!videoWriter.isOpened())
	{
		cout << "Video writer not opening" << endl;
		return -1;
	}*/

	//Mat edges;
	Mat frame;
	Mat result;
	namedWindow("Camera", 1);
	for (;;)
	{
		for (int j = 0; j < 10; ++j)
			cap >> frame;
		//frame = imread("C:\\Users\\ִלטענטי\\Desktop\\x_53b06582.jpg");
		
		if (frame.empty())
			break;

		imageProcessing(frame, distances, relativePoints);

		if (relativePoints.size())
		{
			for (int i = 0; i < relativePoints.size(); i++)
			{
				/*relP.push_back(relativePoints[i].x);
				relP.push_back(relativePoints[i].y);
				*/
				cv::circle(frame, relativePoints[i], 1, Scalar(255, 0, 0));
			}

			Mat dist(1, distances.size()/*48 * 2*/, CV_32FC1, &distances[0]);
			
			model->predict(dist, result);
			index = -1;
			maxResult = -1;
			//cout << result;
			for (int j = 0; j < countEmotion; j++)
			if (result.at<float>(j) > maxResult)
			{
				index = j;
				maxResult = result.at<float>(j);
			}
		}

		for (int i = 0; i < countEmotion; ++i)
		{
			float weight = result.at<float>(i);
			char _weightChar[10];
			weight = (weight < 0) ? 0 : (weight > 1) ? 1 : weight;
			sprintf(_weightChar, "%.2f", weight);
			string outputText = classEmotion[i] + ": " + _weightChar;
			cv::putText(frame, outputText, Point(width - 280, 30 + i * 26), CV_FONT_HERSHEY_SIMPLEX, 1, (i == index) ? Scalar(0, 0, 255) : Scalar(0, 0, 0), 2);
			
			if (i == index)
				cv::circle(frame, Point(width - 292, 30 + i * 26 - 10), 10, Scalar(0, 0, 255), 6);
		}

		//videoWriter.write(frame);
		imshow("Camera", frame);

		if (waitKey(30) >= 0) break;

		distances.clear();
		relativePoints.clear();
		relP.clear();
	}
	//videoWriter.release();
	cap.release();
	cv::waitKey();
	
	return 0;
}

void learn()
{
	Mat trainingData, trainingDataP, trainingDataD;
	Mat labels, labelsP, labelsD;
	std::vector<wstring> filesName;
	ifstream csvFile;
	std::vector<float> points;
	string space;
	int point;

	cout << "Read files" << endl;

	//readCSVCustomDist(trainingDataP, labelsP);

	readCSVDist(trainingData, labelsD);
	//readCSVDist(trainingDataD, labelsD);

	for (int i = 0; i < labelsD.rows; ++i)
	{
		/*Mat tmp(1, trainingDataD.cols + trainingDataP.cols, CV_32FC1);
		for (int j = 0; j < trainingDataD.cols; ++j)
			tmp.at<float>(j) = trainingDataD.row(i).at<float>(j);

		for (int j = trainingDataD.cols; j < trainingDataD.cols + trainingDataP.cols; ++j)
			tmp.at<float>(j) = trainingDataP.row(i).at<float>(j - trainingDataD.cols);
*/
		float classif[countEmotion] = { 0 };
		classif[labelsD.at<int>(i)] = 1.f;
		Mat classificator(1, countEmotion, CV_32F, &classif[0]);

		//trainingData.push_back(tmp);
		labels.push_back(classificator);
	}

	trainingData.convertTo(trainingData, CV_32F);
	labels.convertTo(labels, CV_32F/*CV_32SC1*/);

	Ptr<ml::TrainData> tData = ml::TrainData::create(trainingData, ROW_SAMPLE, labels);

	/* https://github.com/Itseez/opencv/blob/master/samples/cpp/letter_recog.cpp
	http://docs.opencv.org/3.0-beta/modules/ml/doc/neural_networks.html
	http://stackoverflow.com/questions/28709316/neural-network-opencv-3-0 */

	int layer_sz[] = { trainingData.cols, trainingData.cols * 2, countEmotion };
	int nlayers = (int)(sizeof(layer_sz) / sizeof(layer_sz[0]));
	Mat layer_sizes(1, nlayers, CV_32S, layer_sz);

	int method = ANN_MLP::BACKPROP;
	double method_param = 0.0001;
	int max_iter = 10000;

	cout << "Training the classifier (may take a few minutes)...\n";
	Ptr<cv::ml::ANN_MLP> model = ANN_MLP::create();
	model->setLayerSizes(layer_sizes);
	model->setActivationFunction(ANN_MLP::SIGMOID_SYM);
	model->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, max_iter, 1.0e-8));
	model->setTrainMethod(method, method_param);
	model->setBackpropMomentumScale(0.1);
	model->train(tData);

	Mat result;
	model->predict(trainingData, result);

	model->save("ANN_dist_3layers_2.yml");

	//cout << "Load SVM" << endl;
	//// Set up SVM's parameters
	//Ptr<SVM> svm = ml::SVM::create();
	//svm->setType(ml::SVM::C_SVC);
	//svm->setKernel(ml::SVM::POLY);
	//svm->setDegree(3);
	////svm->setGamma(0.001);
	////svm->setTermCriteria(cv::TermCriteria(CV_TERMCRIT_EPS, 100000000, FLT_EPSILON));

	////// Train the SVM
	//cout << "Begin train..." << endl;
	//svm->trainAuto(tData);
	//cout << "End train" << endl;
	////svm->train(trainingData, ml::ROW_SAMPLE, labels);

	//svm->save("Poly3SVMDPoint.yml");
	//
	////svm = Algorithm::load<SVM>("LinerSVC.yml");
	//cout << "Begin predict ..." << endl;

	//Mat result;
	//svm->predict(trainingData, result);

	int errors = 0;

	for (int i = 0; i < trainingData.rows; ++i)
	{
		int index = -1;
		float maxResult = -1;
		for (int j = 0; j < countEmotion; j++)
		if (result.row(i).at<float>(j) > maxResult)
		{
			index = j;
			maxResult = result.row(i).at<float>(j);
		}

		cout << "Class: " + std::to_string(labelsD.at<int>(i)) << " Result: " + std::to_string(index) << endl;
		if (labelsD.at<int>(i) != index) ++errors;
		/*cout << "Class: " + std::to_string(labels.at<int>(i)) << " Result: " + std::to_string(result.at<float>(i)) << endl;
		if ((float)labels.at<int>(i) != result.at<float>(i)) ++errors;*/
	}

	cout << "Errors: " + std::to_string((float)errors / (float)trainingData.rows) + "%" << endl;
	cout << std::to_string(errors) + "/" + std::to_string(trainingData.rows) << endl;
}

void imageProcessing(Mat &cvImg, std::vector<float> &distances, std::vector<Point2f> &relativePoints)
{
	full_object_detection shape;
	array2d<rgb_pixel> img;
	std::vector<dlib::rectangle> dets;
	std::vector<Point2f> points;

	assign_image(img, cv_image<bgr_pixel>(cvImg));
	dets = detector(img);

	for (int i = 0; i < dets.size(); ++i)
	{
		shape = sp(img, dets[i]);

		for (int j = 0; j < shape.num_parts(); ++j)
			points.push_back(Point(shape.part(j).x(), shape.part(j).y()));

		cv::RotatedRect rRect = cv::minAreaRect(points);
		rRect.size = rRect.size + Size2f(2, 2);
		Point2f vertices[4];
		std::map<float, Point2f> tmp;
		float angle;

		rRect.points(vertices);



		for (int i = 0; i < 4; ++i)
		{
			decltype(points[0]) p = vertices[i];
			angle = innerAngle(p, points[27], points[30]);

			tmp.insert(std::pair<float, Point2f>(angle, vertices[i]));
		}

		{
			int i = 0;
			for (auto rec : tmp)
				vertices[i++] = rec.second;
		}

		float w = lengthLine(vertices[0], vertices[1]); // (pow(vertices[0].x - vertices[1].x, 2) + pow(vertices[0].y - vertices[1].y, 2));
		float h = lengthLine(vertices[0], vertices[3]); //sqrt(pow(vertices[0].x - vertices[3].x, 2) + pow(vertices[0].y - vertices[3].y, 2));
		Mat face(h, w, cvImg.type());
		Point2f dvertices[4] = { Point2f(0, 0), Point2f(w, 0), Point2f(w, h), Point2f(0, h) };

		Mat M = cv::getPerspectiveTransform(vertices, dvertices);

		//cv::Homo
		cv::Mat pts_h;

		cv::perspectiveTransform(points, points, M.inv());

		//cv::convertPointsToHomogeneous(points, pts_h);
		//std::cout << M;
		//pts_h = pts_h.reshape(1);
		//pts_h.convertTo(pts_h, CV_64F);
		//pts_h = M.inv()*pts_h;
		//pts_h.convertTo(pts_h, CV_32F);
		//cv::convertPointsFromHomogeneous(pts_h,points);
		////for (auto &p : points)
		////	p *= cvImg.cols;

		cv::warpPerspective(cvImg, face, M, face.size());

		assign_image(img, cv_image<bgr_pixel>(face));
		shape = sp(img, dlib::rectangle(h, w));
		points.clear();

		for (int j = 0; j < shape.num_parts(); ++j)
		{
			points.push_back(Point2f(shape.part(j).x(), shape.part(j).y()));
			//cv::circle(face, points[j], 2, Scalar(50, 0, 255));
		}

		for (int i = 0; i < points.size(); i++)
			relativePoints.push_back(Point2f((float)points[i].x / w, (float)points[i].y / h));

		for (int i = 0; i < points.size(); i++)
			distances.push_back(lengthLine(relativePoints[30], relativePoints[i]));

		relativePoints.erase(relativePoints.begin() + 27, relativePoints.begin() + 30);
		relativePoints.erase(relativePoints.begin(), relativePoints.begin() + 17);

		distances.erase(distances.begin() + 27, distances.begin() + 30);
		distances.erase(distances.begin(), distances.begin() + 17);
	}
}

void readCSVPoint(Mat &trainingData, Mat &labels)
{
	std::vector<float>	 points;
	std::vector<wstring> filesName;

	for (int i = 1; i <= countEmotion; ++i)
	{
		getAllFilesInDir(emotionSort + L"\\" + std::to_wstring(i), filesName);

		for (int j = 0; j < filesName.size(); ++j)
		{
			if (filesName[j].find(L".csv") == std::string::npos || filesName[j].find(L"_dist.csv") != std::string::npos) continue;

			readData(filesName[j], ";", points);

			points.erase(points.begin() + 27 * 2, points.begin() + 30 * 2);
			points.erase(points.begin(), points.begin() + 17 * 2);

			Mat tmp(1, 48 * 2, CV_32FC1, &points[0]);
			trainingData.push_back(tmp);
			labels.push_back(i - 1);

			points.clear();
		}

		filesName.clear();
	}
}

void readCSVCustomDist(Mat &trainingData, Mat &labels)
{
	std::vector<float>	 points;
	std::vector<wstring> filesName;
	std::vector<Point2f> points2f;

	for (int i = 1; i <= countEmotion; ++i)
	{
		getAllFilesInDir(emotionSort + L"\\" + std::to_wstring(i), filesName);

		for (int j = 0; j < filesName.size(); ++j)
		{
			if (filesName[j].find(L".csv") == std::string::npos || filesName[j].find(L"_dist.csv") != std::string::npos) continue;

			readData(filesName[j], ";", points);

			for (int i = 0; i < points.size() - 1; i += 2)
			points2f.push_back(Point2f(points[i], points[i + 1]));

			points.clear();

			points.push_back(lengthLine(points2f[36], points2f[39]));
			points.push_back(lengthLine(points2f[37], points2f[41]));
			points.push_back(lengthLine(points2f[38], points2f[40]));

			points.push_back(lengthLine(points2f[42], points2f[45]));
			points.push_back(lengthLine(points2f[43], points2f[47]));
			points.push_back(lengthLine(points2f[44], points2f[46]));

			for (int i = 0; i < 3; ++i)
			{
			points.push_back(lengthLine(points2f[37], points2f[18 + i]));
			points.push_back(lengthLine(points2f[38], points2f[20 + i]));
			points.push_back(lengthLine(points2f[43], points2f[22 + i]));
			points.push_back(lengthLine(points2f[44], points2f[24 + i]));
			}

			points.push_back(lengthLine(points2f[37], points2f[17]));
			points.push_back(lengthLine(points2f[44], points2f[26]));

			for (int i = 0; i < 4; ++i)
			points.push_back(lengthLine(points2f[61 + i], points2f[67 - i]));

			for (int i = 0; i < 6; ++i)
			{
			points.push_back(lengthLine(points2f[49 + i], points2f[59 - i]));
			points.push_back(lengthLine(points2f[30], points2f[31 + i]));
			}

			points.push_back(lengthLine(points2f[60], points2f[64]));


			Mat tmp(1, 37, CV_32FC1, &points[0]);
			trainingData.push_back(tmp);
			labels.push_back(i - 1);

			points.clear();
		}

		filesName.clear();
	}
}

void readCSVDist(Mat &trainingData, Mat &labels)
{
	std::vector<float>	 points;
	std::vector<wstring> filesName;

	for (int i = 1; i <= countEmotion; ++i)
	{
		getAllFilesInDir(emotionSort + L"\\" + std::to_wstring(i), filesName);

		for (int j = 0; j < filesName.size(); ++j)
		{
			if (filesName[j].find(L"_dist\.csv") == std::string::npos) continue;

			readData(filesName[j], ";", points);
			points.erase(points.begin() + 27, points.begin() + 30);
			points.erase(points.begin(), points.begin() + 17);

			Mat tmp(1, 48, CV_32FC1, &points[0]);
			trainingData.push_back(tmp);
			labels.push_back(i - 1);

			points.clear();
		}

		filesName.clear();
	}
}

void SortAndLandmarkImgs()
{
	wstring srcFullPathImg, destFullPathImg, csvPath;
	std::vector<wstring> allLabels, parseName;
	ifstream infile;
	ofstream outfile;
	int category;
	frontal_face_detector detector = get_frontal_face_detector();
	//shape_predictor sp;
	full_object_detection shape;
	array2d<rgb_pixel> img;
	std::locale loc("rus");
	std::string tmpPath, csvTmp, tmpFileName;
	std::vector<dlib::rectangle> dets;

	deserialize("C:\\Users\\ִלטענטי\\Desktop\\dlib-18.18\\shape_predictor_68_face_landmarks.dat") >> sp;
	Mat imgM;
	for (int i = 1; i < 9; i++)
	{
		//getAllFilesInDir(csvPath + L"C:\\Users\\ִלטענטי\\Desktop\\extended-cohn-kanade-images\\Test\\" + std::to_wstring(i), allLabels);
		getAllFilesInDir(emotionTest + L"\\" + std::to_wstring(i), allLabels);

		int countImg = 0;
		for (int j = 0; j < allLabels.size(); ++j)
		{
			if (allLabels[j].find(L".png") != std::string::npos)
			{
				parseName = parseFileName(allLabels[j]);
				//srcFullPathImg = emotionSort + L"\\" + std::to_wstring(category) + L"\\" + parseName[3] + L"\.png";;
				/*destFullPathImg = emotionTest + L"\\" + std::to_wstring(i) + L"\\" + parseName[3] + L"_flip.png";

				string tmp = wstringToString(allLabels[j], loc);

				imgM = imread(tmp);
				cv::flip(imgM, imgM, 1);

				tmp = wstringToString(destFullPathImg, loc);
				imwrite(tmp, imgM);*/

				size_t indexType = allLabels[j].find_last_of(L".");

				wstring fileName = allLabels[j].substr(0, indexType);
				
				srcFullPathImg = emotionImgs + L"\\" + parseName[0] + L"\\" + parseName[1] + L"\\" + parseName[3] + L"\.png";
				destFullPathImg = emotionNormalized + L"\\" + std::to_wstring(i);
				csvPath = emotionSort + L"\\" + std::to_wstring(i) + L"\\" + parseName[3];// + L"\.csv";
				csvPath = parseName[3] + L".png";

				tmpPath = wstringToString(allLabels[j], loc);
				csvTmp = wstringToString(destFullPathImg, loc);
				tmpFileName = wstringToString(csvPath, loc);

				imgM = imread(tmpPath);

				allocationFaceComponent(imgM, tmpFileName, csvTmp);

				cout << j << "/" << allLabels.size() << "\r";
				
				countImg++;
			}
				

			//continue;
		}

		allLabels.clear();

		cout << "Class: " << i << " images: " << countImg << endl;
	}
	return;

	//deserialize("C:\\Users\\ִלטענטי\\Desktop\\dlib-18.18\\shape_predictor_68_face_landmarks.dat") >> sp;
	//Mat imgM;
	//for (int i = 1; i < 9; i++)
	//{
	//	//getAllFilesInDir(csvPath + L"C:\\Users\\ִלטענטי\\Desktop\\extended-cohn-kanade-images\\Test\\" + std::to_wstring(i), allLabels);
	//	getAllFilesInDir(emotionTest + L"\\" + std::to_wstring(i), allLabels);

	//	int countImg = 0;
	//	for (int j = 0; j < allLabels.size(); ++j)
	//	{
	//		if (allLabels[j].find(L".png") != std::string::npos)
	//		{
	//			parseName = parseFileName(allLabels[j]);
	//			//srcFullPathImg = emotionSort + L"\\" + std::to_wstring(category) + L"\\" + parseName[3] + L"\.png";;
	//			/*destFullPathImg = emotionTest + L"\\" + std::to_wstring(i) + L"\\" + parseName[3] + L"_flip.png";

	//			string tmp = wstringToString(allLabels[j], loc);

	//			imgM = imread(tmp);
	//			cv::flip(imgM, imgM, 1);

	//			tmp = wstringToString(destFullPathImg, loc);
	//			imwrite(tmp, imgM);*/

	//			size_t indexType = allLabels[j].find_last_of(L".");

	//			wstring fileName = allLabels[j].substr(0, indexType);
	//			
	//			srcFullPathImg = emotionImgs + L"\\" + parseName[0] + L"\\" + parseName[1] + L"\\" + parseName[3] + L"\.png";
	//			destFullPathImg = emotionSort + L"\\" + std::to_wstring(i) + L"\\" + parseName[3] + L"\.png";
	//			csvPath = emotionSort + L"\\" + std::to_wstring(i) + L"\\" + parseName[3];// + L"\.csv";

	//			tmpPath = wstringToString(allLabels[j], loc);
	//			csvTmp = wstringToString(fileName, loc);

	//			Landmarks lm;
	//			drawLandmark(tmpPath, &lm, csvTmp, true);

	//			cout << j << "/" << allLabels.size() << "\r";
	//			
	//			countImg++;
	//		}
	//			

	//		//continue;
	//	}

	//	allLabels.clear();

	//	cout << "Class: " << i << " images: " << countImg << endl;
	//}
	//return;
	wstring examplePath = L"C:\\Users\\ִלטענטי\\Desktop\\002";
	getAllFilesInDir(examplePath, allLabels);
	int countImg = 0;
	std::vector<Landmarks> _lms;
	for (int i = 0; i < allLabels.size(); ++i)
	{
		/*if (allLabels[i].find(L".png") != std::string::npos)
			countImg++;

			continue;*/
		parseName = parseFileName(allLabels[i]);

		infile.open(allLabels[i]);
		infile >> category;
		infile.close();

		srcFullPathImg = emotionImgs + L"\\" + parseName[0] + L"\\" + parseName[1] + L"\\" + parseName[3] + L"\.png";
		destFullPathImg = emotionSort + L"\\" + std::to_wstring(category) + L"\\" + parseName[3] + L"\.png";
		csvPath = emotionSort + L"\\" + std::to_wstring(category) + L"\\" + parseName[3];// + L"\.csv";

		wstring labelTxt = emotionImgs + L"\\" + parseName[0] + L"\\" + parseName[1] + L"\\" + std::to_wstring(category) + L".txt";
		outfile.open(labelTxt, std::ofstream::trunc);
		outfile << category;
		outfile.close();

		//copyFile(srcFullPathImg, destFullPathImg);

		tmpPath = wstringToString(allLabels[i], loc);
		csvTmp = wstringToString(csvPath, loc);

		Landmarks lm;
		//drawLandmark(tmpPath, &lm, csvTmp, true);
		//drawLandmark(tmpPath, &lm);
		_lms.push_back(lm);

		cout << i << "/" << allLabels.size() << "\r";

		/*load_image(img, tmpPath);
		dets = detector(img);
		shape = sp(img, dets[0]);

		ofstream csvFile;
		tmpPath = wstringToString(csvPath, loc);
		csvFile.open(tmpPath);

		for (int i = 17; i < shape.num_parts(); ++i)
		csvFile << shape.part(i).x() << ";" << shape.part(i).y() << endl;

		csvFile.close();*/
	}

	ofstream _dist, _point;
	_dist.open(examplePath + L"\\dist.csv");
	_point.open(examplePath + L"\\point.csv");
	for (int i = 0; i < _lms.size(); i++)
	{
		for (int j = 0; j < _lms[i].dist.size(); j++)
		{
			_dist << _lms[i].dist[j] << ";";//(j == (int)_lms[i].dist.size() - 1) ? "" : ";";
			_point << _lms[i].rel[j].x << ";" << _lms[i].rel[j].y << ";";//(j == (int)_lms[i].dist.size() - 1) ? "" : ";";
		}

		_dist << endl;
		_point << endl;
	}

	_dist.close();
	_point.close();

	cout << countImg;
}

//void drawLandmark(string &imgName, string &outFileName, std::vector<float> &dist, std::vector<Point2f> &rel, bool writeCSV /*= false*/)
void drawLandmark(Mat imgName, Landmarks *lm /*= new Landmarks()*/, string outFileName /*= ""*/, bool writeCSV /*= false*/)
{
	frontal_face_detector detector = get_frontal_face_detector();
	//shape_predictor sp;
	full_object_detection shape;
	array2d<rgb_pixel> img;
	std::vector<dlib::rectangle> dets;
	Mat cvImg;
	std::vector<Point2f> points;

	//deserialize("C:\\Users\\ִלטענטי\\Desktop\\dlib-18.18\\shape_predictor_68_face_landmarks.dat") >> sp;

	cvImg = imgName;//imread(imgName);
	assign_image(img, cv_image<bgr_pixel>(cvImg));
	dets = detector(img);

	for (int i = 0; i < dets.size(); ++i)
	{
		shape = sp(img, dets[i]);
		
		for (int j = 0; j < shape.num_parts(); ++j)
			points.push_back(Point(shape.part(j).x(), shape.part(j).y()));
			
		cv::RotatedRect rRect = cv::minAreaRect(points);
		rRect.size = rRect.size;
		Point2f vertices[4];
		std::map<float, Point2f> tmp;
		float angle;

		rRect.points(vertices);
		for (int i = 0; i < 4; ++i)
		{
			//line(cvImg, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 0));
			//cout << vertices[i];

			angle = innerAngle(vertices[i], points[27], points[30]);

			//cout << angle << endl;
			
			tmp.insert(std::pair<float, Point2f>(angle, vertices[i]));
		}

		{
			int i = 0;
			for (auto rec : tmp)
				vertices[i++] = rec.second;
		}

		float w = lengthLine(vertices[0], vertices[1]); // (pow(vertices[0].x - vertices[1].x, 2) + pow(vertices[0].y - vertices[1].y, 2));
		float h = lengthLine(vertices[0], vertices[3]); //sqrt(pow(vertices[0].x - vertices[3].x, 2) + pow(vertices[0].y - vertices[3].y, 2));
		Mat face(h, w, cvImg.type());
		Point2f dvertices[4] = { Point2f(0, 0), Point2f(w, 0), Point2f(w, h), Point2f(0, h) };

		Mat M = cv::getPerspectiveTransform(vertices, dvertices);
		cv::warpPerspective(cvImg, face, M, face.size());

		assign_image(img, cv_image<bgr_pixel>(face));
		shape = sp(img, dlib::rectangle(h, w));
		points.clear();

		for (int j = 0; j < shape.num_parts(); ++j)
		{
			points.push_back(Point2f(shape.part(j).x(), shape.part(j).y()));
			//cv::circle(face, points[j], 2, Scalar(50, 0, 255));
		}

		std::vector<float> distances;
		std::vector<Point2f> relativePoints;

		for (int i = 0; i < points.size(); i++)
		{
			float x, y;
			x = (float)points[i].x / w;
			y = (float)points[i].y / h;
			x = (x < 0.0f) ? 0.0f : ((x > 1.0f) ? 1.0f : x);
			y = (y < 0.0f) ? 0.0f : ((y > 1.0f) ? 1.0f : y);

			relativePoints.push_back(Point2f(x, y));
		}

		for (int i = 0; i < points.size(); i++)
			distances.push_back(lengthLine(relativePoints[30], relativePoints[i]));

		if (writeCSV)
		{
			ofstream relativCSV, destCSV;
			string fileName;

			fileName = outFileName + "\.csv";
			relativCSV.open(fileName, std::ofstream::trunc);

			fileName = outFileName + "_dist\.csv";
			destCSV.open(fileName, std::ofstream::trunc);

			for (int i = 0; i < points.size(); i++)
			{
				relativCSV << relativePoints[i].x << ";" << relativePoints[i].y << endl;
				destCSV << distances[i] << endl;
			}

			relativCSV.close();
			destCSV.close();
		}
		
		//imshow("Roi", face);

		lm->dist = distances;
		lm->rel = relativePoints;
	}

	//

	//imshow("Rectangle", cvImg);
}

void splitString(const string &fullstr, std::vector<string> &elements, const string &delimiter) {

	string::size_type lastpos =
		fullstr.find_first_not_of(delimiter, 0);
	string::size_type pos =
		fullstr.find_first_of(delimiter, lastpos);

	while ((string::npos != pos) || (string::npos != lastpos)) {

		elements.push_back(fullstr.substr(lastpos, pos - lastpos));

		lastpos = fullstr.find_first_not_of(delimiter, pos);
		pos = fullstr.find_first_of(delimiter, lastpos);
	}
}

float stringToFloat(const string &str) {

	istringstream stm;
	float val = 0;

	stm.str(str);
	stm >> val;

	return val;
}

void readData(const wstring &filename,
	const string &csvdelimiter, std::vector< float > &sarr) {

	ifstream fin(filename.c_str());

	string s;
	std::vector<string> selements;

	while (!fin.eof()) {

		getline(fin, s);

		if (!s.empty()) {

			splitString(s, selements, csvdelimiter);

			for (size_t i = 0; i < selements.size(); i++) {
				sarr.push_back(stringToFloat(selements[i]));
			}

			selements.clear();
		}
	}

	fin.close();
}

std::string wstringToString(std::wstring const &s, std::locale const &loc, char default_char)
{
	if (s.empty())
		return std::string();
	std::ctype<wchar_t> const &facet = std::use_facet<std::ctype<wchar_t> >(loc);
	wchar_t const *first = s.c_str();
	wchar_t const *last = first + s.size();
	std::vector<char> result(s.size());

	facet.narrow(first, last, default_char, &result[0]);

	return std::string(result.begin(), result.end());
}

void getAllFilesInDir(wstring strPath, std::vector<wstring> & strFiles)
{
	WIN32_FIND_DATA fd; HANDLE handle;
	wstring strSpec = strPath + L"\\*.*";
	handle = FindFirstFile(strSpec.c_str(), &fd);
	if (handle == INVALID_HANDLE_VALUE)
		return;
	do {
		strSpec = fd.cFileName;
		if (strSpec != L"." && strSpec != L"..")
		{
			if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_HIDDEN))
			{
				strSpec = strPath + L"\\" + strSpec;
				if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
					getAllFilesInDir(strSpec, strFiles);
				}
				else {
					if (fd.nFileSizeLow != 0 || fd.nFileSizeHigh != 0) {
						strFiles.push_back(strSpec);
					}
				}
			}
		}
	} while (FindNextFile(handle, &fd));
	FindClose(handle);
}

std::vector<wstring> parseFileName(wstring fullfileName)
{
	std::vector<wstring> parseFileName;

	size_t found = fullfileName.find_last_of(L"\\");

	wstring fileName = fullfileName.substr(found + 1, fullfileName.length());

	parseFileName.push_back(fileName.substr(0, 4));
	parseFileName.push_back(fileName.substr(5, 3));
	parseFileName.push_back(fileName.substr(9, 8));
	parseFileName.push_back(fileName.substr(0, fileName.find_last_of('.')));

	return parseFileName;
}

void copyFile(const std::wstring& fileNameFrom, const std::wstring& fileNameTo)
{
	std::ifstream in(fileNameFrom.c_str(), std::ios::binary);
	std::ofstream out(fileNameTo.c_str(), std::ios::binary);
	out << in.rdbuf();
	out.close();
	in.close();
}

template<class T>
T innerAngle(cv::Point_<T> &p1, cv::Point_<T> &p2, cv::Point_<T> &c)
{
	auto a = p1 - c;
	auto b = p2 - c;

	auto A = atan2(a.y, a.x) - atan2(b.y, b.x) * 180 / CV_PI;

	return A;
}

float lengthLine(cv::Point2f &p1, cv::Point2f &p2)
{
	return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

void parseCSV(wstring fileName)
{
	std::vector<Point2f> frame;
	std::vector<std::vector<Point2f>> points;
	string csvdelimiter = ";";
	string s;
	std::vector<string> selements;
	
	ifstream fin(fileName.c_str());	

	while (!fin.eof()) {

		getline(fin, s);

		if (!s.empty()) {

			splitString(s, selements, csvdelimiter);

			for (size_t i = 0; i < selements.size(); i+=2) {
				frame.push_back(Point2f(stringToFloat(selements[i]), stringToFloat(selements[i + 1])));
			}

			points.push_back(frame);

			selements.clear();
			frame.clear();
		}
	}

	//Mat marker(600, 600, CV_32F);
	//for (int i = 0; i < points.size(); ++i)
	//{
	//	for (int j = 0; j < points[i].size(); ++j)
	//		break;
	//		//cv::putText(marker, std::to_string(j), points[i][j] * 1000, cv::HersheyFonts::FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(255, 0, 0));
	//}
	//float xMean = 0, yMean = 0;
	//for (int i = 0; i < points[0].size(); ++i)
	//{
	//	for (int j = 0; j < points.size(); ++j)
	//	{
	//		xMean += points[j][i].x;
	//		yMean += points[j][i].y;
	//	}

	//	xMean /= (float)points.size();
	//	yMean /= (float)points.size();

	//	//cv::circle(marker, Point2f(xMean, yMean) * 1000, 2, Scalar(0, 0, 255));
	//	cv::putText(marker, std::to_string(i), Point2f(xMean, yMean) * 500 + Point2f(50, 50), cv::HersheyFonts::FONT_HERSHEY_COMPLEX_SMALL, 0.5, Scalar(255, 0, 0));

	//	xMean = 0;
	//	yMean = 0;
	//}
	//cv::cvtColor(marker, marker, CV_GRAY2BGR);
	//fin.close();

	//imshow("Marker", marker);
	//waitKey();
}

void imagesToVideo()
{
	wstring emotionPath = L"C:\\Work\\Surprise video";
	cv::VideoWriter _videoWriter("surprise.avi", CV_FOURCC('D', 'I', 'V', 'X'), 5, Size(640, 490));
	std::vector<wstring> files;
	std::locale loc("rus");
	Mat imgg;
	std::vector<Landmarks> _lms;
	namedWindow("Video");
	namedWindow("Point");

	for (int i = 1; i <= 2; i++)
	{
		getAllFilesInDir(emotionPath + L"\\Surprise " + std::to_wstring(i), files);

		for (int j = 0; j < files.size(); ++j)
		{
			Mat _pointImg(300, 300, CV_32F, double(0));
			Landmarks lm;
			string _tmpStr = wstringToString(files[j], loc);
			imgg = imread(_tmpStr);

			drawLandmark(imgg, &lm);
			_lms.push_back(lm);

			for (int z = 0; z < lm.rel.size(); ++z)
			{
				cv::circle(_pointImg, Point2f(lm.rel[z].x, lm.rel[z].y) * 250 + Point2f(30, 30), 2, Scalar(255, 0, 0), 2);
			}

			imshow("Video", imgg);
			imshow("Point", _pointImg);
			_videoWriter.write(imgg);
			cv::waitKey(1);
		}

		files.clear();
	}

	ofstream _dist, _point;
	_dist.open(emotionPath + L"\\dist.csv");
	_point.open(emotionPath + L"\\point.csv");
	for (int i = 0; i < _lms.size(); i++)
	{
		for (int j = 0; j < _lms[i].dist.size(); j++)
		{
			_dist << _lms[i].dist[j] << ";";//(j == (int)_lms[i].dist.size() - 1) ? "" : ";";
			_point << _lms[i].rel[j].x << ";" << _lms[i].rel[j].y << ";";//(j == (int)_lms[i].dist.size() - 1) ? "" : ";";
		}

		_dist << endl;
		_point << endl;
	}

	_dist.close();
	_point.close();

	_videoWriter.release();
}

void markingVideo()
{
	wstring srcFullPathImg, destFullPathImg, csvPath;
	std::vector<wstring> allLabels, parseName;
	ifstream infile;
	ofstream outfile;
	int category;
	frontal_face_detector detector = get_frontal_face_detector();
	full_object_detection shape;
	array2d<rgb_pixel> img;
	std::locale loc("rus");
	std::string tmpPath, csvTmp;
	std::vector<dlib::rectangle> dets;
	Mat frame;

	wstring examplePath = L"C:\\Work";
	getAllFilesInDir(examplePath, allLabels);
	int countImg = 0;
	std::vector<Landmarks> _lms;

	cv::VideoCapture cap("C:\\Work\\surprise.avi"); // open the default camera
	if (!cap.isOpened())  // check if we succeeded
	{
		cout << "Cap not opening" << endl;
		return;
	}

	namedWindow("Video");

	while (cap.read(frame))
	{
		Landmarks lm;
		//drawLandmark(frame, &lm);

		//_lms.push_back(lm);

		imshow("Video", frame);
	} 

	ofstream _dist, _point;
	_dist.open(examplePath + L"\\dist.csv");
	_point.open(examplePath + L"\\point.csv");
	for (int i = 0; i < _lms.size(); i++)
	{
		for (int j = 0; j < _lms[i].dist.size(); j++)
		{
			_dist << _lms[i].dist[j] << ";";//(j == (int)_lms[i].dist.size() - 1) ? "" : ";";
			_point << _lms[i].rel[j].x << ";" << _lms[i].rel[j].y << ";";//(j == (int)_lms[i].dist.size() - 1) ? "" : ";";
		}

		_dist << endl;
		_point << endl;
	}

	_dist.close();
	_point.close();
}

int main1()
{
	cout << "Load frontal face detector" << endl;
	detector = get_frontal_face_detector();

	cout << "Load shape predictor" << endl;
	deserialize("C:\\Work\\shape_predictor_68_face_landmarks.dat") >> sp;

	cout << "Start marking video" << endl;
	imagesToVideo();
	//markingVideo();
	cout << "End marking video" << endl;

	_getch();
	
	return 0;
}

void allocationFaceComponent(Mat imgName, string imgFileName, string outFileName)
{
	frontal_face_detector detector = get_frontal_face_detector();
	//shape_predictor sp;
	full_object_detection shape;
	array2d<rgb_pixel> img;
	std::vector<dlib::rectangle> dets;
	Mat cvImg;
	std::vector<Point2f> points;

	//deserialize("C:\\Users\\ִלטענטי\\Desktop\\dlib-18.18\\shape_predictor_68_face_landmarks.dat") >> sp;

	cvImg = imgName;//imread(imgName);
	assign_image(img, cv_image<bgr_pixel>(cvImg));
	dets = detector(img);

	for (int i = 0; i < dets.size(); ++i)
	{
		shape = sp(img, dets[i]);

		for (int j = 0; j < shape.num_parts(); ++j)
			points.push_back(Point(shape.part(j).x(), shape.part(j).y()));

		string dirFace = outFileName + "\\Face\\" + imgFileName;
		string dirEye = outFileName + "\\Eye\\" + imgFileName;
		string dirMouth = outFileName + "\\Mouth\\" + imgFileName;
		string dirNose = outFileName + "\\Nose\\" + imgFileName;

		Mat eye, nose, mouth;
		std::vector<Point2f> tmp_point;

		for (int i = 48; i <= 67; i++)
			tmp_point.push_back(points[i]);

		mouth = cvImg(boundingRect(tmp_point));
		tmp_point.clear();

		for (int i = 27; i <= 35; i++)
			tmp_point.push_back(points[i]);

		nose = cvImg(boundingRect(tmp_point));
		tmp_point.clear();

		for (int i = 17; i <= 21; i++)
			tmp_point.push_back(points[i]);
		for (int i = 22; i <= 26; i++)
			tmp_point.push_back(points[i]);
		for (int i = 36; i <= 41; i++)
			tmp_point.push_back(points[i]);
		for (int i = 42; i <= 47; i++)
			tmp_point.push_back(points[i]);

		eye = cvImg(boundingRect(tmp_point));
		tmp_point.clear();

		
		imwrite(dirEye, eye);
		imwrite(dirMouth, mouth);
		imwrite(dirNose, nose);

		cv::RotatedRect rRect = cv::minAreaRect(points);
		rRect.size = rRect.size;
		Point2f vertices[4];
		std::map<float, Point2f> tmp;
		float angle;

		rRect.points(vertices);
		for (int i = 0; i < 4; ++i)
		{
			//line(cvImg, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 0));
			//cout << vertices[i];

			angle = innerAngle(vertices[i], points[27], points[30]);

			//cout << angle << endl;

			tmp.insert(std::pair<float, Point2f>(angle, vertices[i]));
		}

		{
			int i = 0;
			for (auto rec : tmp)
				vertices[i++] = rec.second;
		}

		float w = lengthLine(vertices[0], vertices[1]); // (pow(vertices[0].x - vertices[1].x, 2) + pow(vertices[0].y - vertices[1].y, 2));
		float h = lengthLine(vertices[0], vertices[3]); //sqrt(pow(vertices[0].x - vertices[3].x, 2) + pow(vertices[0].y - vertices[3].y, 2));
		Mat face(h, w, cvImg.type());
		Point2f dvertices[4] = { Point2f(0, 0), Point2f(w, 0), Point2f(w, h), Point2f(0, h) };

		Mat M = cv::getPerspectiveTransform(vertices, dvertices);
		cv::warpPerspective(cvImg, face, M, face.size());

		assign_image(img, cv_image<bgr_pixel>(face));
		shape = sp(img, dlib::rectangle(h, w));
		points.clear();

		for (int j = 0; j < shape.num_parts(); ++j)
		{
			float x = shape.part(j).x() > w ? w : (shape.part(j).x() < 0 ? 0 : shape.part(j).x());
			float y = shape.part(j).y() > h ? h : (shape.part(j).y() < 0 ? 0 : shape.part(j).y());
			points.push_back(Point2f(x, y));
			//cv::circle(face, points[j], 2, Scalar(50, 0, 255));
		}

		imwrite(dirFace, face);
	}

	//

	//imshow("Rectangle", cvImg);
}