// human_detect.cpp

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/video.hpp> // for BackgroundSubtractorMOG2
#include <opencv2/opencv.hpp>
#include "Hungarian.h"

using namespace cv;
using namespace std;

double BG_BACKGROUNDRATIO; // maximum percentage of background (1-c_f)
double BG_CRTHRESHOLD; // complexity reduction threshold (minimum description length) (c_T)
bool BG_SHADOW; // shadow detection and removal
int BG_HISTORY; // learning rate (1/BG_HISTORY)
int BG_NMIXTURES; // maximum number of Gaussians
double BG_SHADOWTHRESHOLD; // relative brightness of shadow
int BG_SHADOWVALUE; // value assigned to shadow
double BG_VARINIT; // initial variance
double BG_VARMAX; // maximum variance
double BG_VARMIN; // minimum variance
double BG_VARTHRESHOLD; // threshold to declare background
double BG_VARTHRESHOLDGEN; // threshold for a new component
double BG_LEARNRATE; // learning rate (1 = no history used, 0 = no update)


int MORPH_ITR; // number of iterations for morphological operations
int MORPH_OPEN_RADIUS; // remove noise in foreground masks
int MORPH_CLOSE_RADIUS; // remove noise in foreground masks

int OBJ_MAX_AREA_THRESHOLD; // maximum area of detected object
int OBJ_MIN_AREA_THRESHOLD; // minimum area of detected object

const int BUFFER_SIZE = 150; // number of frames to store

bool DISPLAY_FPS; // display FPS
bool VISUALIZE; // visualize results
bool WRITE_OUTPUT; // write results into a video file

bool CONVERT_2_GREY; // convert raw frame to greyscale to speedup processing
double RESIZE_FACTOR; // scale factor for image resize
bool SET_ROI; // allow user to select a ROI
bool ROTATE_90_CCW; // rotate video 90 degree counter-clockwise

const int TC_SIZE = 100; // buffer for estimating FPS



bool set_roi_done = false;
bool first_objects = true;

int idgenerator = 0;
double PROCESSNOISECOV;
double MEASUREMENTNOISECOV;
double ERRORCOVPOST;
int INVISIBLEFORTOOLONG;
int AGETHRESHOLD;
int MINIMUM_AGE;
int MINIMUM_CHANGE_IN_X;
int MINIMUM_TOTAL_VISIBLE_COUNT;

struct track {
	int id;
	Rect bbox;
	KalmanFilter Kalman = KalmanFilter(4, 2, 0);
	int age;
	int totalVisibleCount;
	int consecutiveInvisibleCount;
	int x_max;
	double x_min;
	int change_in_x;
	Mat predictedcenteroid;
};
vector<Point> areay;
vector<Mat> filteredCentroids;
vector<Rect> bboxs;
vector<Point> matches;
vector<Point> pairs;
vector<int> unassignedDetections;
vector<int> unassignedTracks;
vector<track> tracks;
vector<int> unassignedbboxids;
vector<int> matchedbboxids;

void help() {
	cout <<
		"Road crossing human detection\n"
		"Usage: human_detect <video_file_path> [arguments]\n\n"
		"Press q, Q, or <esc> to quit\n\n"
		"Arguments:\n"
		"  --BG_BACKGROUNDRATIO=<double>\n"
		"    Minimum percentage of background.\n"
		"    Affects classification into fg/bg and estimated bg intensity. Default 0.99.\n"
		"  --BG_CRTHRESHOLD=<double>\n"
		"    Complexity reduction threshold. Default 0.05.\n"
		"  --BG_SHADOW=<bool>\n"
		"    Shadow detection and removal. Default false.\n"
		"  --BG_HISTORY=<int>\n"
		"    Learning rate (1/BG_HISTORY).\n"
		"    Overwritten by BG_LEARNRATE. Default 200.\n"
		"  --BG_NMIXTURES=<int>\n"
		"    Maxmimum number of Gaussians. Default 5.\n"
		"  --BG_SHADOWTHRESHOLD=<double>\n"
		"    Relative brightness of shadow. Default 0.6.\n"
		"  --BG_SHADOWVALUE=<int>\n"
		"    Value assigned to shadow. Default 127.\n"
		"  --BG_VARINIT=<int>\n"
		"    Initial variance. Default 15.0.\n"
		"  --BG_VARMAX=<double>\n"
		"    Maximum variance. Default 75.0.\n"
		"  --BG_VARMIN=<double>\n"
		"    Minimum variance. Default 4.0.\n"
		"  --BG_VARTHRESHOLD=<double>\n"
		"    Threshold to declare background.\n"
		"    Affects classification into fg/bg and estimated bg intensity. Default 16.0.\n"
		"  --BG_VARTHRESHOLDGEN=<double>\n"
		"    Threshold for a new component. Default 9.0.\n"
		"  --BG_LEARNRATE=<double>\n"
		"    Learning rate (1 = no history used, 0 = no update). Default 0.001.\n"
		"  --MORPH_ITR=<int>\n"
		"    Number of iterations for morphological operations. Default 1.\n"
		"  --MORPH_DILATION_RADIUS=<int>\n"
		"    Remove noise in foreground masks. Default 1.\n"
		"  --DISPLAY_FPS=<bool>\n"
		"    Display estimated FPS. Default false.\n"
		"  --VISUALIZE=<bool>\n"
		"    Visualize results. Default true.\n"
		"  --WRITE_OUTPUT=<bool>\n"
		"    Write results into a video file. Default false.\n"
		"  --CONVERT_2_GREY=<bool>\n"
		"    Convert raw frame to greyscale to speedup processing.\n"
		"  --RESIZE_FACTOR=<double>\n"
		"    Scale factor for image resize. Default 1.0.\n"
		"  --SET_ROI=<bool>\n"
		"    Prompt user to select a ROI. Default is false.\n"
		"    Click to select points. Double-click to add a point and stop. Max 100 points.\n"
		"  --ROTATE_90_CCW=<bool>\n"
		"    Rotate video 90 degree counter-clockwise. Default is false.\n"
		"  --PROCESSNOISECOV = <double>\n"
		"	 Process noise covariance matrix. Default is .005.\n"
		"  --MEASUREMENTNOISECOV = <double>\n"
		"	 Measurement noise covariance matrix. Default is 1e-1.\n"
		"  --ERRORCOVPOST = <double>\n"
		"	 Posteriori error estimate covariance matrix. Default is 0.1.\n"
		"  --INVISIBLEFORTOOLONG = <int>\n"
		"	 Threshold for frames object is invisble. Default is 20.\n"
		"  --AGETHRESHOLD = <int>\n"
		"	 Age threshold for track deletion. Default is 8.\n"
		"  --MINIMUM_AGE = <int>\n"
		"	 Miumium age for human to be shown. Default is 60.\n"
		"  --MINIMUM_CHANGE_IN_X = <int>\n"
		"	 Miumium horizontal distnce traveled by human to be shown. Default is 75.\n"
		"  --MINIMUM_TOTAL_VISIBLE_COUNT = <int>\n"
		"	 Miumium total visible count by human to be shown. Default is 25.\n"
		"  --help\n"
		"    Print help.\n";
}


void initializeTrack(const float x_CurrentCentroid, const float y_CurrentCentroid, Rect &BBox)//done
{
	track inittrack;
	cout << "Initializing Track" << endl;
	inittrack.id = idgenerator++;
	inittrack.bbox.x = BBox.x;
	inittrack.bbox.y = BBox.y;
	inittrack.bbox.height = BBox.height;
	inittrack.bbox.width = BBox.width;
	cout << "Initializing Kalman Filter" << endl;
	inittrack.Kalman.init(4, 2, 0);

	float F[] = { 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1 };
	inittrack.Kalman.transitionMatrix = cv::Mat(4, 4, CV_32F, F).clone();// Transition Matrix
	cout << "current centroids " << x_CurrentCentroid << " " << y_CurrentCentroid << endl;

	inittrack.Kalman.statePost.setTo(0);
	inittrack.Kalman.statePost.at<float>(0, 0) = x_CurrentCentroid;
	inittrack.Kalman.statePost.at<float>(1, 0) = y_CurrentCentroid;


	setIdentity(inittrack.Kalman.measurementMatrix);
	setIdentity(inittrack.Kalman.processNoiseCov, Scalar::all(PROCESSNOISECOV));
	setIdentity(inittrack.Kalman.measurementNoiseCov, Scalar::all(MEASUREMENTNOISECOV));
	setIdentity(inittrack.Kalman.errorCovPost, Scalar::all(ERRORCOVPOST));
	cout << "Kalman Filter initialized" << endl;

	cout << "State " << inittrack.Kalman.statePost << endl;

	inittrack.age = 1;
	inittrack.totalVisibleCount = 1;
	inittrack.consecutiveInvisibleCount = 0;
	inittrack.x_max = (int)x_CurrentCentroid;
	inittrack.x_min = (int)x_CurrentCentroid;
	inittrack.change_in_x = 0;
	tracks.push_back(inittrack);
}

double distance(const double x1, const double y1, const double x2, const double y2) //done
{
	return sqrt((pow(x1 - x2, 2) + pow(y1 - y2, 2)));
}
// Update assigned tracks
void updateAssignedTracks()  //done
{
	if (matches.size() == 0)
	{
		cout << "No tracks updated " << endl;
		return;
	}
	for (int i = 0; i < matches.size(); i++)
	{
		Mat pass(2, 1, CV_32F);
		cout << "updating matched tracks" << endl;
		int trackidx = matches.at(i).x;
		int detectionidx = matches.at(i).y;
		pass.at<float>(0) = filteredCentroids.at(detectionidx).at<double>(0);
		pass.at<float>(1) = filteredCentroids.at(detectionidx).at<double>(1);
		Mat estimated = tracks.at(trackidx).Kalman.correct(pass);
		Point statePt(estimated.at<float>(0), estimated.at<float>(1));
		cout << "corrected centroid " << statePt << endl;
		cout << "state Pt " << statePt << endl;
		tracks.at(trackidx).bbox.x = statePt.x;
		tracks.at(trackidx).bbox.y = statePt.y;
		tracks.at(trackidx).age++;
		tracks.at(trackidx).totalVisibleCount++;
		tracks.at(trackidx).consecutiveInvisibleCount = 0;
		cout << "statePt.x " << statePt.x << endl;
		if ((int)statePt.x <= tracks.at(trackidx).x_min)
		{
			tracks.at(trackidx).x_min = (int)statePt.x;
		}
		if ((int)statePt.x >= tracks.at(trackidx).x_max)
		{
			tracks.at(trackidx).x_max = (int)statePt.x;
		}
		cout << " x.max " << tracks.at(trackidx).x_max << "x.min " << tracks.at(trackidx).x_min << endl;
		tracks.at(trackidx).change_in_x = tracks.at(trackidx).x_max - tracks.at(trackidx).x_min;
		cout << "change in x " << tracks.at(trackidx).change_in_x << endl;
	}
	cout << "Tracks updated " << endl;
}

// Update unassigned tracks
void updateUnassignedTracks() //done
{
	if (unassignedTracks.size() == 0)
	{
		cout << "no unassigned Tracks" << endl;
		return;
	}
	cout << "update unassigned Tracks" << endl;
	for (int i = 0; i < unassignedTracks.size(); i++)
	{
		int index = unassignedTracks.at(i);
		tracks.at(index).age++;
		tracks.at(index).consecutiveInvisibleCount++;
	}
	cout << "unassigned tracks updated" << endl;
}

// Delete lost tracks
void deleteLostTracks()//done
{
	cout << "deleting lost track" << endl;

	if (tracks.size() == 0)
	{
		cout << "tracks empty" << endl;
		return;
	}

	for (vector<track>::iterator it = tracks.begin(); it != tracks.end();)
	{
		double visibility = (double)it->totalVisibleCount / (double)it->age;
		if (((it->age < AGETHRESHOLD) && (visibility < 0.6)) || it->consecutiveInvisibleCount >= INVISIBLEFORTOOLONG)
		{
			tracks.erase(it);
			cout << "deleted lost track" << endl;
		}
		else
		{
			it++;
		}
	}
}

// Create new tracks for unassigned detected centroids
void createNewTracks()//done
{
	if (unassignedDetections.size() == 0)
	{
		cout << "no unassigned detections" << endl;
		return;
	}
	cout << "creating new track" << endl;
	for (int i = 0; i < unassignedDetections.size(); i++)
	{
		initializeTrack(filteredCentroids.at(unassignedDetections.at(i)).at<double>(0), filteredCentroids.at(unassignedDetections.at(i)).at<double>(1), bboxs.at(unassignedDetections.at(i)));
	}
	cout << "created new track" << endl;
}

void detectionToTrackAssignment()//done
{
	pairs.clear();
	unassignedDetections.clear();
	unassignedTracks.clear();
	matches.clear();
	matchedbboxids.clear();
	unassignedbboxids.clear();

	cout << "main porcess" << endl;
	int paircounter = 0;
	int length = tracks.size() + filteredCentroids.size();
	cout << "length " << length << endl;
	vector< vector<double> > costMatrix(length);
	//initialize 2d vector
	for (int i = 0; i < length; i++)
	{
		// declare  the i-th row to size of column
		costMatrix[i] = vector<double>(length);
		for (int j = 0; j < length; j++)
			costMatrix[i][j] = 0;
	}

	HungarianAlgorithm HungAlgo;
	vector<int> assignment;

	if (filteredCentroids.size() != 0 && tracks.size() != 0)
	{
		for (int n = 0; n < tracks.size(); n++)
		{
			Mat predict = tracks.at(n).Kalman.predict();
			Point prediction(predict.at<float>(0), predict.at<float>(1));
			tracks.at(n).bbox.x = prediction.x;
			tracks.at(n).bbox.y = prediction.y;
			cout << "Prediction " << prediction << endl;
			for (int m = 0; m < filteredCentroids.size(); m++)
			{
				costMatrix.at(n).at(m) = distance(filteredCentroids.at(m).at<double>(0), filteredCentroids.at(m).at<double>(1), prediction.x, prediction.y);
				cout << "detected and distance " << filteredCentroids.at(m).at<double>(0) << " " << filteredCentroids.at(m).at<double>(1) << " " << costMatrix.at(n).at(m) << endl;
			}
		}
		for (int n = 0; n < length; n++)
		{
			for (int m = 0; m < length; m++)
			{
				if ((n >= tracks.size()) && (m < filteredCentroids.size()))
				{
					costMatrix.at(n).at(m) = 20;
				}
				else if ((n < tracks.size()) && (m >= filteredCentroids.size()))
				{
					costMatrix.at(n).at(m) = 20;
				}
				else if ((n >= tracks.size()) && (m >= filteredCentroids.size()))
				{
					costMatrix.at(n).at(m) = 0;
				}
			}
		}
		cout << "Matrix formed" << endl;
		for (int n = 0; n < length; n++)
		{
			for (int m = 0; m < length; m++)
			{
				cout << costMatrix.at(n).at(m) << " ";
			}
			cout << endl;
		}

		double cost = HungAlgo.Solve(costMatrix, assignment);
		cout << "Matrix Solved" << endl;

		for (unsigned int x = 0; x < costMatrix.size(); x++)
		{
			cout << x << "," << assignment[x] << "\t";
		}
		cout << endl;
		for (int i = 0; i < costMatrix.size(); i++)
		{

			Point pairpoint(i, assignment[i]);
			pairs.push_back(pairpoint);
			cout << "pairs " << pairs.at(paircounter)<<endl;
			paircounter++;
		}
		cout << "Pairs made" << endl;
		int matchcount = 0;
		for (int i = 0; i < pairs.size(); i++)
		{
			cout << "detections and tracks present" << endl;
			if ((pairs.at(i).x >= tracks.size()) && (pairs.at(i).y < filteredCentroids.size()))
			{
				cout << "unassigned detections present "<<" pairs.at(i)"<< pairs.at(i)<< endl;
				unassignedDetections.push_back(pairs.at(i).y);
			}
			else if ((pairs.at(i).x  < tracks.size()) && (pairs.at(i).y >= filteredCentroids.size()))
			{
				cout << "unassigned tracks present" << " pairs.at(i)" << pairs.at(i) << endl;
				unassignedTracks.push_back(pairs.at(i).x);
			}
			else if (pairs.at(i).x < tracks.size() && pairs.at(i).y < filteredCentroids.size())
			{
				cout << "Matching " << endl;
				matches.push_back(pairs.at(i));
				cout << " pairs.at(i)" << pairs.at(i) << endl;
				cout << "match size " << matchcount << " match pairs " << matches.at(matchcount++) << endl;
			}
		}
	}
	if (filteredCentroids.size() == 0 && tracks.size() != 0)
	{
		cout << "no detections but tracks are there" << endl;
		for (int i = 0; i < tracks.size(); i++)
		{
			unassignedTracks.push_back(i);
		}
	}
	if (filteredCentroids.size() != 0 && tracks.size() == 0)
	{
		cout << "Emtpy tracks but detections are present" << endl;
		for (int i = 0; i < filteredCentroids.size(); i++)
		{
			unassignedDetections.push_back(i);
		}
	}
	updateAssignedTracks();
	updateUnassignedTracks();
	deleteLostTracks();
	createNewTracks();
	cout << "Everything completed " << "Next frame" << endl;
	cout << "End check of track size " << tracks.size() << endl;
	for (int i = 0; i < tracks.size(); i++)
	{
		cout << "Track info " << "id " << tracks.at(i).id << " age " << tracks.at(i).age << " total visible count " << tracks.at(i).totalVisibleCount << " consecutiveInvisibleCount " << tracks.at(i).consecutiveInvisibleCount << endl;
	}
}

void onMouse(int event, int x, int y, int flags, void* param)
{
	static Point PointArray[1][100];

	static int roi_np = 0;

	if ((event == EVENT_LBUTTONDOWN) && (roi_np <= 99))
	{
		PointArray[0][roi_np] = Point(x, y);
		roi_np++;
		//cout << "Point #" << roi_np << ": " << PointArray[0][roi_np - 1] << endl;
	}

	if (((event == EVENT_LBUTTONDBLCLK) && roi_np >= 3) || (roi_np >= 100)) // Selection is done
	{
		int roi_np_arr[1] = { roi_np };
		Scalar color(255, 255, 255);
		const Point* ppt[1] = { PointArray[0] };

		Mat& ROI = *(Mat*)param;

		Mat ROI_lines(ROI.size(), CV_8UC3, Scalar(0, 0, 0));
		polylines(ROI_lines, ppt, roi_np_arr, 1, true, color);
		imshow("SetROI", ROI_lines);
		Mat ROI2(ROI.size(), CV_8UC3, Scalar(0, 0, 0));
		fillPoly(ROI2, ppt, roi_np_arr, 1, color);
		cvtColor(ROI2, ROI2, COLOR_BGR2GRAY);
		threshold(ROI2, ROI, 128, 255, THRESH_BINARY);

		set_roi_done = true;
	}
}


int process(string input_video)
{
	// Initialize video input
	VideoCapture capture(input_video); //try to open a video file or image sequence with a string
	if (!capture.isOpened())
	{
		capture.open(atoi(input_video.c_str()) + cv::CAP_DSHOW); //try to open a camera with an integer
	}
	if (!capture.isOpened()) {
		cerr << "Failed to open the video device, video file or image sequence!\n" << endl;
		help();
		return EXIT_FAILURE;
	}


	/* Get video frame size */
	const int frame_width = static_cast<int>(capture.get(ROTATE_90_CCW ? CAP_PROP_FRAME_HEIGHT : CAP_PROP_FRAME_WIDTH));
	const int frame_height = static_cast<int>(capture.get(ROTATE_90_CCW ? CAP_PROP_FRAME_WIDTH : CAP_PROP_FRAME_HEIGHT));
	const int frame_resized_width = (int)(frame_width * RESIZE_FACTOR);
	const int frame_resized_height = (int)(frame_height * RESIZE_FACTOR);

	// Get the first frame for ROI selection
	Mat frame, frame_resized;
	capture >> frame;
	if (frame.empty())
		return EXIT_FAILURE;
	capture.set(CAP_PROP_POS_FRAMES, 0); // rewind
	if (ROTATE_90_CCW) rotate(frame, frame, ROTATE_90_COUNTERCLOCKWISE);
	resize(frame, frame_resized, cv::Size(frame_resized_width, frame_resized_height));


	/* Set ROI */
	Mat ROI(frame_resized.size(), CV_8U);
	ROI.setTo(255);
	if (SET_ROI)
	{
		namedWindow("SetROI", WINDOW_AUTOSIZE);//check this
		imshow("SetROI", frame_resized);
		setMouseCallback("SetROI", onMouse, (void*)&ROI);
		while (!set_roi_done && (char)waitKey(10) != 'q')
		{
			// do nothing
		}
		waitKey(1000);
		destroyWindow("SetROI");
	}

	/* Initialize display windows */
	string frame_window_name = "Video frame";
	string obj_window_name = "Moving objects";

	if (VISUALIZE)
	{
		namedWindow(frame_window_name, WINDOW_AUTOSIZE);
		namedWindow(obj_window_name, WINDOW_AUTOSIZE);
	}

	//cout << "Press <q> or <esc> to quit, <space> to toggle learning rate update method" << endl;
	cout << "Press <q> or <esc> to quit" << endl;

	/* Initialize background estimators */
	Ptr<BackgroundSubtractorMOG2> pMOG; //MOG2 Background subtractor
	pMOG = createBackgroundSubtractorMOG2();
	//pMOG = createBackgroundSubtractorMOG2(BG_HISTORY, BG_VARTHRESHOLD, BG_SHADOW);

	pMOG->setBackgroundRatio(BG_BACKGROUNDRATIO);
	pMOG->setComplexityReductionThreshold(BG_CRTHRESHOLD);
	pMOG->setDetectShadows(BG_SHADOW);
	pMOG->setHistory(BG_HISTORY);
	pMOG->setNMixtures(BG_NMIXTURES);
	pMOG->setShadowThreshold(BG_SHADOWTHRESHOLD);
	pMOG->setShadowValue(BG_SHADOWVALUE);
	pMOG->setVarInit(BG_VARINIT);
	pMOG->setVarMax(BG_VARMAX);
	pMOG->setVarMin(BG_VARMIN);
	pMOG->setVarThreshold(BG_VARTHRESHOLD);
	pMOG->setVarThresholdGen(BG_VARTHRESHOLDGEN);

	//bool update_bg_auto = false; // automatically adjust learning rate or fix it

	Mat frame_reduced;
	Mat frame_dilated; //dilation
	Mat bg; // bg estimated by MOG2
	Mat fgMask; //fg estimated by MOG2

				// The first fgMask obtained by MOG2 is often all foreground, instead of all background.
				// Thus, we first let MOG2 learn repeatedly the initial frame for a few times.
	if (CONVERT_2_GREY)
		cvtColor(frame_resized, frame_reduced, COLOR_BGR2GRAY);
	else
		frame_reduced = frame_resized;
	for (int i = 0; i<10; i++)
		pMOG->apply(frame_reduced, fgMask, BG_LEARNRATE);


	/* Structural elements for removing noise in estimated masks */
	Mat element_open = getStructuringElement(MORPH_ELLIPSE,   //MORPH_RECT
		Size(2 * MORPH_OPEN_RADIUS + 1, 2 * MORPH_OPEN_RADIUS + 1),
		Point(MORPH_OPEN_RADIUS, MORPH_OPEN_RADIUS)); // for removing noise

	Mat element_close = getStructuringElement(MORPH_ELLIPSE,
		Size(2 * MORPH_CLOSE_RADIUS + 1, 2 * MORPH_CLOSE_RADIUS + 1),
		Point(MORPH_CLOSE_RADIUS, MORPH_CLOSE_RADIUS)); // for removing noise

														/* Statistics of connected components for moving objects*/
	Mat labels, stats, centroids;
	int NumComp, NumComp2;

	/* Buffer for replaying */
	Mat frame_buffer[BUFFER_SIZE], fgMask_buffer[BUFFER_SIZE];
	for (int i = 0; i < BUFFER_SIZE; i++) {
		frame_buffer[i] = Mat::zeros(frame_resized.size(), frame.type());
		fgMask_buffer[i] = Mat::zeros(frame_resized.size(), CV_8U);
	}

	int frame_id;
	int frame_buffer_index = 0;

	double tc[TC_SIZE];
	int tc_index = 0;

	vector<cv::Mat> planes(3);

	// For displaying video frame juxtaposed with bounding boxes of detected objects
	Mat frame_box = Mat::zeros(frame_resized.size(), frame.type());

	// For displaying frame and foreground mask in one big window
	Mat big_window = Mat::zeros(cv::Size(frame_resized_width * 2, frame_resized_height), frame.type());

	// Write the video frames (with bounding boxes) into file
	VideoWriter output;
	if (WRITE_OUTPUT)
	{
		output.open("outputVideo.avi", CV_FOURCC('D', 'I', 'V', 'X'), 25, frame_box.size(), true);
	}

	char key;
	Mat LookUpTable;

	// Main loop
	for (;;) {
		tc[tc_index] = (double)getTickCount();
		// Acquire, rotate, convert to greyscale, and resize
		frame_id = static_cast<int>(capture.get(CAP_PROP_POS_FRAMES));
		//cout << "Frame ID: " << frame_id << endl;
		capture >> frame;
		if (frame.empty())
			return EXIT_FAILURE;
		if (ROTATE_90_CCW)
			rotate(frame, frame, ROTATE_90_COUNTERCLOCKWISE);
		resize(frame, frame_resized, cv::Size(frame_resized_width, frame_resized_height));
		if (CONVERT_2_GREY)
			cvtColor(frame_resized, frame_reduced, COLOR_BGR2GRAY);
		else
			frame_reduced = frame_resized;
		frame_resized.copyTo(frame_buffer[frame_buffer_index]);

		// Update the background model (fgMask has 1 channel with value 0 or 255)
		pMOG->apply(frame_reduced, fgMask, BG_LEARNRATE);
		//pMOG->apply(frame_reduced, fgMask, update_bg_auto ? -1 : BG_LEARNRATE);

		// Get the background (overall average of Gaussian mixture, possibly multichannel)
		//pMOG->getBackgroundImage(bg);

		// Clean up the noise
		morphologyEx(fgMask, fgMask, cv::MORPH_OPEN, element_open, Point(-1, -1), MORPH_ITR);
		morphologyEx(fgMask, fgMask, cv::MORPH_CLOSE, element_close, Point(-1, -1), MORPH_ITR);

		// Exclude object pixels outside ROI (to speed up the labelling of connected components)
		fgMask = fgMask & ROI;

		// Filter objects with large size
		NumComp = connectedComponentsWithStats(fgMask, labels, stats, centroids, 8, CV_32S) - 1; // Subtract 1 to exclude background

																								 // Map label of connected components to 0 or 255
																								 // 0 for bg and objects of size too small or too large
																								 // 255 for accepted objects
		filteredCentroids.clear();
		bboxs.clear();

		dilate(frame_reduced, frame_dilated, Mat());
		if (NumComp == 0) 
		{
			NumComp2 = 0;
		}
		else
		{ 
			unsigned char *p_lut, *p_fg;
			int *p_lbl; // labels has 4 bytes
						// Construct the map
			LookUpTable = Mat::zeros(1, NumComp + 1, CV_8U);
			p_lut = LookUpTable.ptr();
			p_lut[0] = 0;
			for (int k = 1; k <= NumComp; k++)
			{
				int a = stats.at<int>(k, CC_STAT_AREA);
				OBJ_MAX_AREA_THRESHOLD = 900;
				OBJ_MIN_AREA_THRESHOLD = 55;
				p_lut[k] = ((a <= OBJ_MAX_AREA_THRESHOLD) && (a >= OBJ_MIN_AREA_THRESHOLD)) ? 255 : 0;
				
				if (p_lut[k] > 0)
				{
					Mat center(1, 2, CV_32F);
					center.at<double>(0) = stats.at<int>(k, 0) - 5;
					center.at<double>(1) = stats.at<int>(k, 1) - 5;
					filteredCentroids.push_back(center);
					cout << endl;
					cout << "Road crossing human is found at frame #" << frame_id << ". Locations:" << center.at<double>(0) << " " << center.at<double>(1) << endl;
					cout << "stats check " << Point(stats.at<int>(k, 0) - 5, stats.at<int>(k, 1) - 5) << endl;
					Rect box(stats.at<int>(k, 0) - 5, stats.at<int>(k, 1) - 5, stats.at<int>(k, 2) + 10, stats.at<int>(k, 3) + 10);
					bboxs.push_back(box);
				}

			}
			cout << "All centroids detected and stored " << filteredCentroids.size() << " Bbox size " << bboxs.size() <<" track size "<<tracks.size()<< endl;

			// Reset fgMask to 0 for those removed connected components
			for (int i = 0; i < frame_resized_height; i++) {
				p_fg = fgMask.ptr<unsigned char>(i);
				p_lbl = labels.ptr<int>(i);
				for (int j = 0; j < frame_resized_width; j++) {
					p_fg[j] = p_lut[p_lbl[j]];
				}
			}
			NumComp2 = countNonZero(LookUpTable);
		}

		fgMask.copyTo(fgMask_buffer[frame_buffer_index]);
		detectionToTrackAssignment();
		// Overlay bounding boxes on video frame

		frame_resized.copyTo(frame_box);
		for (int i = 0; i < bboxs.size(); i++)
		{
			cout << "bboxs change in x " << tracks.at(i).change_in_x;
			if ((tracks.at(i).change_in_x > MINIMUM_CHANGE_IN_X) && (tracks.at(i).age > MINIMUM_AGE) && (tracks.at(i).totalVisibleCount > MINIMUM_TOTAL_VISIBLE_COUNT))
			{
				tracks.at(i).bbox.width = bboxs.at(i).width;
				tracks.at(i).bbox.height = bboxs.at(i).height;
				rectangle(frame_box, tracks.at(i).bbox, Scalar(0, 0, 255), 2);
			}
		}


		// Visualize results
		if (VISUALIZE)
		{
			imshow(frame_window_name, frame_box);
			imshow(obj_window_name, fgMask);

			//cvtColor(fgMask, fgMask, COLOR_GRAY2BGR);
			//frame_box.copyTo(big_window(Rect(0, 0, frame_resized_width, frame_resized_height)));
			//fgMask.copyTo(big_window(Rect(frame_resized_width, 0, frame_resized_width, frame_resized_height)));
		}


		key = (char)waitKey(1); // delay n milliseconds to allow time for display

		switch (key) {
		case 'q':
		case 'Q':
		case 27: //escape key
			return EXIT_SUCCESS;
			//case ' ':
			//	update_bg_auto = !update_bg_auto;
			//	cout << "Auto update background model = " << update_bg_auto << endl;
		default:
			break;
		}

		/* move to next buffer in a cyclic way */
		frame_buffer_index = (++frame_buffer_index) % BUFFER_SIZE;
		tc_index = (++tc_index) % TC_SIZE;

		if (DISPLAY_FPS)
			cout << endl << "Frame #" << frame_id <<
			", FPS " << ((frame_id >= TC_SIZE ? TC_SIZE : frame_id)*getTickFrequency()) / ((double)getTickCount() - tc[tc_index]) << endl;

		if (WRITE_OUTPUT)
		{
			output << frame_box;
			//output << big_window;
		}
	}
	return EXIT_SUCCESS;
}


int main(int argc, char** argv)
{
	const char *keys =
		"{ @input                       |        | input video file or image sequence }"
		"{ BG_BACKGROUNDRATIO           | 0.70   | minimum percentage of background }"
		"{ BG_CRTHRESHOLD               | 0.05   | complexity reduction threshold }"
		"{ BG_SHADOW                    | false  | shadow detection and removal }"
		"{ BG_HISTORY                   | 200    | learning rate (1/BG_HISTORY) }"
		"{ BG_NMIXTURES                 | 5      | maxmimum number of Gaussians }"
		"{ BG_SHADOWTHRESHOLD           | 0.6    | relative brightness of shadow }"
		"{ BG_SHADOWVALUE               | 127    | value assigned to shadow }"
		"{ BG_VARINIT                   | 15.0   | initial variance }"
		"{ BG_VARMAX                    | 75.0   | maximum variance }"
		"{ BG_VARMIN                    | 4.0    | minimum variance }"
		"{ BG_VARTHRESHOLD              | 16.0   | threshold to declare background }"
		"{ BG_VARTHRESHOLDGEN           | 9.0    | threshold for a new component }"
		"{ BG_LEARNRATE                 | 0.005  | learning rate (1 = no history used, 0 = no update) }"
		"{ MORPH_ITR                    | 1      | number of iterations for morphological operations }"
		"{ MORPH_OPEN_RADIUS            | 1      | remove noise in foreground masks }"
		"{ MORPH_CLOSE_RADIUS           | 7      | remove noise in foreground masks }"
		"{ DISPLAY_FPS                  | false  | display estimated FPS }"
		"{ VISUALIZE                    | true   | visualize results }"
		"{ WRITE_OUTPUT                 | false  | write reqsults into a video file }"
		"{ CONVERT_2_GREY               | true   | convert raw frame to greyscale to speedup processing }"
		"{ RESIZE_FACTOR                | 1.0    | scale factor for image resize }"
		"{ SET_ROI                      | false  | prompt user to select a ROI }"
		"{ ROTATE_90_CCW                | false  | rotate video 90 degree counter-clockwise }"
		"{ PROCESSNOISECOV              | .005   | process noise covariance matrix  }"
		"{ MEASUREMENTNOISECOV          | 1e-1   | measurement noise covariance matrix }"
		"{ ERRORCOVPOST                 | 0.1    | posteriori error estimate covariance matrix }"
		"{ INVISIBLEFORTOOLONG          | 20     | Threshold for frames object is invisble }" 
		"{ AGETHRESHOLD                 | 8      | Age threshold for track deletion }"
		"{ MINIMUM_AGE                  | 60     | Miumium age for human to be shown }"
		"{ MINIMUM_CHANGE_IN_X          | 75     | Miumium horizontal distnce traveled by human to be shown }"
		"{ MINIMUM_TOTAL_VISIBLE_COUNT  | 25     | Miumium total visible count by human to be shown }"
		"{ help ||}";

	CommandLineParser parser(argc, argv, keys);

	if (parser.has("help"))
	{
		help();
		return EXIT_SUCCESS;
	}

	string input_video = parser.get<string>("@input");
	if (input_video.empty()) {
		cout << "no input..." << endl;
		help();
		return EXIT_FAILURE;
	}

	BG_BACKGROUNDRATIO = parser.get<double>("BG_BACKGROUNDRATIO");
	BG_CRTHRESHOLD = parser.get<double>("BG_CRTHRESHOLD");
	BG_SHADOW = parser.get<bool>("BG_SHADOW");
	BG_HISTORY = parser.get<int>("BG_HISTORY");
	BG_NMIXTURES = parser.get<int>("BG_NMIXTURES");
	BG_SHADOWTHRESHOLD = parser.get<double>("BG_SHADOWTHRESHOLD");
	BG_SHADOWVALUE = parser.get<int>("BG_SHADOWVALUE");
	BG_VARINIT = parser.get<double>("BG_VARINIT");
	BG_VARMAX = parser.get<double>("BG_VARMAX");
	BG_VARMIN = parser.get<double>("BG_VARMIN");
	BG_VARTHRESHOLD = parser.get<double>("BG_VARTHRESHOLD");
	BG_VARTHRESHOLDGEN = parser.get<double>("BG_VARTHRESHOLDGEN");
	BG_LEARNRATE = parser.get<double>("BG_LEARNRATE");

	MORPH_ITR = parser.get<int>("MORPH_ITR");
	MORPH_OPEN_RADIUS = parser.get<int>("MORPH_OPEN_RADIUS");
	MORPH_CLOSE_RADIUS = parser.get<int>("MORPH_CLOSE_RADIUS");

	DISPLAY_FPS = parser.get<bool>("DISPLAY_FPS");
	VISUALIZE = parser.get<bool>("VISUALIZE");
	WRITE_OUTPUT = parser.get<bool>("WRITE_OUTPUT");

	CONVERT_2_GREY = parser.get<bool>("CONVERT_2_GREY");
	RESIZE_FACTOR = parser.get<double>("RESIZE_FACTOR");
	SET_ROI = parser.get<bool>("SET_ROI");
	ROTATE_90_CCW = parser.get<bool>("ROTATE_90_CCW");
	PROCESSNOISECOV = parser.get<double>("PROCESSNOISECOV");
	MEASUREMENTNOISECOV = parser.get<double>("MEASUREMENTNOISECOV");
	ERRORCOVPOST = parser.get<double>("ERRORCOVPOST");
	INVISIBLEFORTOOLONG = parser.get<int>("INVISIBLEFORTOOLONG");
	AGETHRESHOLD = parser.get<int>("AGETHRESHOLD");

	MINIMUM_AGE = parser.get<int>("MINIMUM_AGE");
	MINIMUM_CHANGE_IN_X = parser.get<int>("MINIMUM_CHANGE_IN_X");
	MINIMUM_TOTAL_VISIBLE_COUNT = parser.get<int>("MINIMUM_TOTAL_VISIBLE_COUNT");

	return process(input_video);
}