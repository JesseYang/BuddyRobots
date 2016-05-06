using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System.Collections.Generic;
using System.Threading;

using OpenCVForUnity;
using System;

namespace OpenCVForUnitySample
{


    public class ARGame : MonoBehaviour
    {

        const int MAX_OBJS = 10;
        const int FRAME_NUMBER = 20;
        const int MIN_PTS_NUM = 100;

        public GameObject applePreFab;
        public GameObject housePreFab;
        public GameObject treePreFab;

        private Track[] m_Tracks;
        private Mat m_Gray;
        private Mat m_Binary;
        private Mat m_Color;
        private Thread m_ThreadDetectFig;
        private Thread m_ThreadDetectCard;

        private bool m_DetectFigure;
        private bool m_DetectCard;
        private bool m_DetectDigit;
        private bool m_DetectFig;
        private bool m_DetectSquare;
        private Mutex m_Mutex;
        private BaseParams m_Params;

        /***** for DetectChange *****/
        private Mat m_lastBinary;
        private Mat m_Temp;
        private Scalar m_Diff;
        private int m_DilateType;
        private int m_DilateSize;
        private Mat m_DilateElement;
        private double[] m_DiffAry;
        private int m_FrameIndex;
        private bool m_Stationary;
        /****************************/

        /**** for cards recgonition ****/
        private Card m_Card;
        private List<int> m_RecognizedCards;
        private List<List<Point>> m_Squares;
        private int m_CardNum;
        private string m_CardObjType;
        /*******************************/

        /**** for digit recgonition ****/
        private string m_DigitUrl = "http://117.121.26.21:5000/";
        public GameObject m_TestImage;
        private Texture2D m_TestImageTexture;
        Color32[] m_TestImageColors;
        List<Point> m_DigitPtList;
        public Text digitResult;
        /*******************************/

        /**** for character create ****/
        public ToggleGroup m_ToggleGroup;
        private Mat m_FigToSubmit;
        /******************************/

        WebCamTexture webCamTexture;
        WebCamDevice webCamDevice;

        Color32[] colors;

        public bool shouldUseFrontFacing = true;

        int width = 1280;
        int height = 720;

        Mat rgbaMat;

        int p1 = 9;
        int p2 = 3;
        bool m_Flip;

        /**** for Frame ****/
        List<Point> m_Frame;
        Mat m_FrameMat;
        Mat m_Homography;
        bool m_ChangeView;
        List<GameObject> m_Apples;
        /*******************/

        Texture2D texture;

        bool initDone = false;

        ScreenOrientation screenOrientation = ScreenOrientation.Unknown;

        void Start()
        {
            shouldUseFrontFacing = true;
            StartCoroutine(init());
        }

        private IEnumerator init()
        {
            Debug.Log(Application.persistentDataPath);
            if (webCamTexture != null)
            {
                webCamTexture.Stop();
                initDone = false;
                rgbaMat.Dispose();
            }

            // Checks how many and which cameras are available on the device
            for (int cameraIndex = 0; cameraIndex < WebCamTexture.devices.Length; cameraIndex++)
            {
                if (WebCamTexture.devices[cameraIndex].isFrontFacing == shouldUseFrontFacing)
                {
                    webCamDevice = WebCamTexture.devices[cameraIndex];
                    webCamTexture = new WebCamTexture(webCamDevice.name, width, height);
                    break;
                }
            }

            if (webCamTexture == null)
            {
                webCamDevice = WebCamTexture.devices[0];
                webCamTexture = new WebCamTexture(webCamDevice.name, width, height);
            }

            webCamTexture.Play();

            m_Tracks = new Track[MAX_OBJS];
            for (int i = 0; i < MAX_OBJS; i++)
            {
                m_Tracks[i] = new Track();
            }
            // m_Params = new BaseParams(640, 480, 13.33333f, 10.0f);
            m_Params = new BaseParams(1080, 720, 17.777778f, 10.0f);
            m_DetectFigure = false;
            m_DetectCard = false;
            m_DetectDigit = false;
            m_DetectSquare = true;
            m_DetectFig = false;
            m_Mutex = new Mutex();

            /**** for DetectChange ****/
            m_lastBinary = new Mat(webCamTexture.height, webCamTexture.width, CvType.CV_8UC4);
            m_DilateType = Imgproc.MORPH_DILATE;
            m_DilateSize = 2;
            m_DilateElement = Imgproc.getStructuringElement(m_DilateType,
                new Size(2 * m_DilateSize + 1, 2 * m_DilateSize + 1),
                new Point(m_DilateSize, m_DilateSize));
            m_FrameIndex = 0;
            m_DiffAry = new double[FRAME_NUMBER];
            m_Stationary = true;
            /**************************/

            /**** for Frame *****/
            m_Frame = new List<Point>(4);
            m_Frame.Add(new Point(240, 30));
            m_Frame.Add(new Point(880, 180));
            m_Frame.Add(new Point(880, 540));
            m_Frame.Add(new Point(240, 690));
            m_FrameMat = new Mat(webCamTexture.height, webCamTexture.width, CvType.CV_8UC4);

            List<Point> points = new List<Point>(4);
            points.Add(new Point(0, 0));
            points.Add(new Point(1280, 0));
            points.Add(new Point(1280, 720));
            points.Add(new Point(0, 720));

            m_Homography = Calib3d.findHomography(new MatOfPoint2f(m_Frame.ToArray()), new MatOfPoint2f(points.ToArray()));
            m_ChangeView = false;
            /**********************/

            /**** for card recgonition ****/
            /*
            m_Card = new Card(CardType.Digit | CardType.Letter);
            m_RecognizedCards = new List<int>();
            m_Squares = new List<List<Point>>();
            m_Apples = new List<GameObject>(9);
            m_CardNum = 0;
            for (int i = 0; i < 9; i++)
            {
                m_Apples.Add(new GameObject());
            }
            */
            /******************************/

            /**** for square detection ****/
            m_Squares = new List<List<Point>>();
            /******************************/

            /**** for digit recgonition ****/
            m_TestImageTexture = new Texture2D(28, 28, TextureFormat.RGBA32, false);
            m_TestImage.GetComponent<Renderer>().material.mainTexture = m_TestImageTexture;
            m_TestImageColors = new Color32[28 * 28];
            m_DigitPtList = new List<Point>();
            /*******************************/

            /**** for figture detect and submit ****/
            m_FigToSubmit = new Mat();
            /***************************************/

            m_Flip = false;

            while (true)
            {
                if (webCamTexture.didUpdateThisFrame)
                {
                    colors = new Color32[webCamTexture.width * webCamTexture.height];
                    rgbaMat = new Mat(webCamTexture.height, webCamTexture.width, CvType.CV_8UC4);
                    m_Gray = new Mat(webCamTexture.height, webCamTexture.width, CvType.CV_8UC4);
                    m_Binary = new Mat(webCamTexture.height, webCamTexture.width, CvType.CV_8UC4);
                    m_Color = new Mat(webCamTexture.height, webCamTexture.width, CvType.CV_8UC4);
                    m_Temp = new Mat(webCamTexture.height, webCamTexture.width, CvType.CV_8UC4);
                    texture = new Texture2D(webCamTexture.width, webCamTexture.height, TextureFormat.RGBA32, false);

                    gameObject.GetComponent<Renderer>().material.mainTexture = texture;

                    screenOrientation = Screen.orientation;
                    initDone = true;
                    break;
                }
                else
                {
                    yield return 0;
                }
            }
        }

        public void onChangeView()
        {
            m_ChangeView = !m_ChangeView;
        }

        void Update()
        {
            if (!initDone)
                return;

#if UNITY_IOS && !UNITY_EDITOR && (UNITY_4_6_3 || UNITY_4_6_4 || UNITY_5_0_0 || UNITY_5_0_1)
            if (webCamTexture.width > 16 && webCamTexture.height > 16) {
#else
            if (webCamTexture.didUpdateThisFrame)
            {
#endif
                Utils.webCamTextureToMat(webCamTexture, rgbaMat, colors);
                /*
                if (!webCamDevice.isFrontFacing)
                {
                    if (webCamTexture.videoRotationAngle == 180)
                    {
                        Core.flip(rgbaMat, rgbaMat, -1);
                    }
                    else if (webCamTexture.videoRotationAngle == 270)
                    {
                        Core.flip(rgbaMat, rgbaMat, -1);
                    }
                }
                else
                {
                    if (webCamTexture.videoRotationAngle == 0)
                    {
                        Core.flip(rgbaMat, rgbaMat, 1);
                    }
                    else if (webCamTexture.videoRotationAngle == 90)
                    {
                        Core.flip(rgbaMat, rgbaMat, 0);
                    }
                    if (webCamTexture.videoRotationAngle == 180)
                    {
                        Core.flip(rgbaMat, rgbaMat, 0);
                    }
                    else if (webCamTexture.videoRotationAngle == 270)
                    {
                        Core.flip(rgbaMat, rgbaMat, 1);
                    }
                }
                */

                if (m_Flip == true)
                    Core.flip(rgbaMat, rgbaMat, 1);

                if (m_ChangeView)
                {
                    Imgproc.cvtColor(m_FrameMat, m_Gray, Imgproc.COLOR_BGR2GRAY);
                }
                else
                {
                    Imgproc.cvtColor(rgbaMat, m_Gray, Imgproc.COLOR_BGR2GRAY);
                }
                // Imgproc.adaptiveThreshold(m_Gray, m_Binary, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, p1, p2);
                Imgproc.threshold(m_Gray, m_Binary, 135, 255, Imgproc.THRESH_BINARY);
                // Imgproc.threshold(m_Gray, m_Binary, 255 * 2.8 / 5, 255, Imgproc.THRESH_BINARY);
                DetectChange();

                if ((m_ThreadDetectFig == null || !m_ThreadDetectFig.IsAlive) && m_DetectFigure == true)
                {
                    m_ThreadDetectFig = new Thread(this.DetectFigure);
                    m_ThreadDetectFig.Start();
                }

                /*
                if ((m_ThreadDetectCard == null || !m_ThreadDetectCard.IsAlive) && m_DetectCard == true)
                {
                    m_ThreadDetectCard = new Thread(this.DetectCard);
                    m_ThreadDetectCard.Start();
                }
                */

                /*
                if (m_DetectFigure == true)
                {
                    DetectFigure();
                }
                */

                if (m_DetectCard == true)
                {
                    DetectCard();
                }

                if (m_DetectSquare == true)
                {
                    DetectSquare();
                }

                if (m_DetectDigit == true)
                {
                    Debug.Log("Detect Digit");
                    DetectDigit();
                }

                if (m_DetectFig == true)
                {
                    Debug.Log("Detect Fig");
                    DetectFig();
                }

                for (int i = 0; i < MAX_OBJS; i++)
                {
                    LoadObj(m_Tracks[i]);
                }

                Imgproc.warpPerspective(rgbaMat, m_FrameMat, m_Homography, new Size());

                m_Mutex.WaitOne();
                if (m_ChangeView)
                {
                    Utils.matToTexture2D(m_FrameMat, texture, colors);
                    // Utils.matToTexture2D(m_Color, texture, colors);
                }
                else
                {
                    Core.line(rgbaMat, m_Frame[0], m_Frame[1], new Scalar(255, 0, 0), 3);
                    Core.line(rgbaMat, m_Frame[1], m_Frame[2], new Scalar(255, 0, 0), 3);
                    Core.line(rgbaMat, m_Frame[2], m_Frame[3], new Scalar(255, 0, 0), 3);
                    Core.line(rgbaMat, m_Frame[3], m_Frame[0], new Scalar(255, 0, 0), 3);
                    Utils.matToTexture2D(rgbaMat, texture, colors);
                    // Utils.matToTexture2D(m_Binary, texture, colors);
                    /*
                    Mat subMat = new Mat(28, 28, CvType.CV_8UC4);
                    subMat = rgbaMat.submat(0, 28, 0, 28);
                    Utils.matToTexture2D(subMat, m_TestImageTexture, m_TestImageColors);
                    */
                }
                m_Mutex.ReleaseMutex();
            }
        }

        void OnDisable()
        {
            webCamTexture.Stop();
        }

        private void LoadObj(Track track)
        {
            if (track.m_Ready == false)
            {
                if (track.m_CurObj != null)
                {
                    Destroy(track.m_CurObj);
                }
                return;
            }
            if (track.m_CurObjType != track.m_objType)
            {
                GameObject targetPreFab;
                switch (track.m_objType)
                {
                    case "apple":
                        targetPreFab = applePreFab;
                        break;
                    case "house":
                        targetPreFab = housePreFab;
                        break;
                    case "tree":
                        targetPreFab = treePreFab;
                        break;
                    default:
                        targetPreFab = applePreFab;
                        break;
                }
                if (track.m_CurObj != null)
                {
                    Destroy(track.m_CurObj);
                }
                track.m_CurObj = Instantiate(targetPreFab, track.m_RPos, Quaternion.identity) as GameObject;
                track.m_CurObj.transform.localScale = new Vector3(track.m_RScale, track.m_RScale, track.m_RScale);
                track.m_CurObjType = track.m_objType;
            }
            if (track.m_CurObjType == track.m_objType)
            {
                track.m_CurObj.transform.position = track.m_RPos;
                track.m_CurObj.transform.localScale = new Vector3(track.m_RScale, track.m_RScale, track.m_RScale);
            }
        }

        private void DetectChange()
        {
            Core.subtract(m_Binary, m_lastBinary, m_Temp);
            Imgproc.erode(m_Temp, m_Temp, m_DilateElement);
            m_Diff = Core.sumElems(m_Temp);
            m_FrameIndex++;
            m_FrameIndex = m_FrameIndex % FRAME_NUMBER;
            m_DiffAry[m_FrameIndex] = m_Diff.val[0];
            double max = 0;
            for (int i = 0; i < FRAME_NUMBER; i++)
            {
                if (m_DiffAry[i] > max)
                    max = m_DiffAry[i];
            }
            if (max > 3000)
            {
                /// motion
                if (m_Stationary)
                {
                    Debug.Log("MOTION!");
                    m_Stationary = false;
                }
            }
            else
            {
                /// motionless
                if (!m_Stationary)
                {
                    Debug.Log("STATIC!");
                    m_Stationary = true;
                }
            }


            m_lastBinary = m_Binary.clone();
        }

        private void DetectFigure()
        {
            if (!m_Stationary)
            {
                return;
            }
            Segmentation seg = new Segmentation(m_Binary);
            List<List<Point>> clusterList = seg.Do();

            int index = 0;
            for (int i = 0; i < clusterList.Count; i++)
            {
                if (index > MAX_OBJS - 1)
                {
                    break;
                }
                if (clusterList[i].Count < MIN_PTS_NUM)
                {
                    continue;
                }
                m_Tracks[index].Calculate(m_Gray.cols(), m_Gray.rows(), clusterList[i], m_Params);
                m_Tracks[index].Update();
                index++;
            }
            for (int i = index; i < MAX_OBJS; i++)
            {
                m_Tracks[i].Clear();
            }
        }

        private void DetectSquare()
        {
            m_Squares.Clear();
            m_Squares = FigProc.FindSquares(m_Binary.clone());
            // m_Squares = FilterSquares(m_Squares);
            for (int i = 0; i < m_Squares.Count; i++)
            {
                Core.line(m_Binary, m_Squares[i][0], m_Squares[i][1], new Scalar(255, 0, 0), 3);
                Core.line(m_Binary, m_Squares[i][1], m_Squares[i][2], new Scalar(255, 0, 0), 3);
                Core.line(m_Binary, m_Squares[i][2], m_Squares[i][3], new Scalar(255, 0, 0), 3);
                Core.line(m_Binary, m_Squares[i][3], m_Squares[i][0], new Scalar(255, 0, 0), 3);
            }
        }

        private void DetectCard()
        {
            m_Squares.Clear();
            m_Squares = m_Card.Recgonize(m_Binary, out m_RecognizedCards, true);
            // m_Mutex.WaitOne();
            Imgproc.cvtColor(m_Binary, m_Color, Imgproc.COLOR_GRAY2BGR);
            for (int i = 0; i < m_Squares.Count; i++)
            {
                Core.line(m_Color, m_Squares[i][0], m_Squares[i][1], new Scalar(255, 0, 0), 3);
                Core.line(m_Color, m_Squares[i][1], m_Squares[i][2], new Scalar(255, 0, 0), 3);
                Core.line(m_Color, m_Squares[i][2], m_Squares[i][3], new Scalar(255, 0, 0), 3);
                Core.line(m_Color, m_Squares[i][3], m_Squares[i][0], new Scalar(255, 0, 0), 3);
            }
            // m_Mutex.ReleaseMutex();
            string str = "";
            for (int i = 0; i < m_RecognizedCards.Count; i++)
            {
                str += (m_RecognizedCards[i] - 0x30) + ", ";
            }

            int index = 0;
            if (m_RecognizedCards.Count > 0)
            {
                m_CardNum = m_RecognizedCards[0] - 0x30;

                float radius = 3.0f;
                for (index = 0; index < m_CardNum; index++)
                {
                    m_Tracks[index].Set(
                        m_CardObjType,
                        radius * Mathf.Cos(index * 1.0f / m_CardNum * 2 * Mathf.PI),
                        radius * Mathf.Sin(index * 1.0f / m_CardNum * 2 * Mathf.PI),
                        50.0f);
                    m_Tracks[index].Update();
                }
            }
            for (int i = index; i < MAX_OBJS; i++)
            {
                m_Tracks[i].Clear();
            }

            Debug.Log(str);
        }

        public void SubmitFig()
        {
            IEnumerable<Toggle> activeToggles = m_ToggleGroup.ActiveToggles();

            Toggle target = null;

            foreach (Toggle toggle in activeToggles)
            {
                Debug.Log(toggle.name);
                target = toggle;
            }

            if (target == null)
            {
                digitResult.text = "请先选择字符";
                return;
            }

            String label = target.name.Split('_')[1];

            FigProc.Img2Pts(m_FigToSubmit, ref m_DigitPtList);
            List<String> ptStrList = new List<string>(m_DigitPtList.Count);
            for (int i = 0; i < m_DigitPtList.Count; i++)
            {
                ptStrList.Add(((27 - m_DigitPtList[i].y) + 28 * m_DigitPtList[i].x).ToString());
            }
            String data = String.Join(",", ptStrList.ToArray());

            WWWForm form = new WWWForm();
            form.AddField("image", data);
            form.AddField("label", label);

            WWW www = new WWW(m_DigitUrl + "create/", form);
            StartCoroutine(WaitForSubmitFigRequest(www));
        }

        IEnumerator WaitForSubmitFigRequest(WWW www)
        {
            yield return www;

            if (www.error == null)
            {
                Debug.Log("WWW OK!: " + www.text);
                digitResult.text = "提交成功";
            }
            else
            {
                Debug.Log("WWW Error: " + www.error);
                digitResult.text = "提交失败，请查看日志";
            }
        }

        private void DetectFig()
        {
            // digit detect only run once
            m_DetectFig = false;

            if (!m_Stationary)
            {
                return;
            }
            Segmentation seg = new Segmentation(m_Binary);
            List<List<Point>> clusterList = seg.Do();

            Mat stdImg = new Mat();
            List<Point> allPts = new List<Point>();
            for (int i = 0; i < clusterList.Count; i++)
            {
                if (clusterList[i].Count < 30)
                {
                    continue;
                }
                else
                {
                    allPts.AddRange(clusterList[i]);
                }
            }
            stdImg = FigProc.ToMnistFormat(allPts);
            Imgproc.threshold(stdImg, stdImg, 200, 255, Imgproc.THRESH_BINARY);
            Utils.matToTexture2D(stdImg, m_TestImageTexture, m_TestImageColors);
            m_FigToSubmit = stdImg;
            digitResult.text = "检测完毕，可以提交"; 
        }

        private void DetectDigit()
        {
            // digit detect only run once
            m_DetectDigit = false;

            if (!m_Stationary)
            {
                return;
            }
            Segmentation seg = new Segmentation(m_Binary);
            List<List<Point>> clusterList = seg.Do();

            Mat stdImg = new Mat();
            bool hasDigit = false;
            for (int i = 0; i < clusterList.Count; i++)
            {
                if (clusterList[i].Count < 30)
                {
                    continue;
                }
                else
                {
                    stdImg = FigProc.ToMnistFormat(clusterList[i]);
                    Imgproc.threshold(stdImg, stdImg, 200, 255, Imgproc.THRESH_BINARY);
                    Utils.matToTexture2D(stdImg, m_TestImageTexture, m_TestImageColors);
                    hasDigit = true;
                    break;
                }
            }

            if (hasDigit)
            {
                FigProc.Img2Pts(stdImg, ref m_DigitPtList);
                List<String> ptStrList = new List<string>(m_DigitPtList.Count);
                for (int i = 0; i < m_DigitPtList.Count; i++)
                {
                    ptStrList.Add(((27 - m_DigitPtList[i].y) + 28 * m_DigitPtList[i].x).ToString());
                }
                String data = String.Join(",", ptStrList.ToArray());
                Debug.Log(data);

                WWWForm form = new WWWForm();
                form.AddField("image", data);

                WWW www = new WWW(m_DigitUrl + "digit/", form);
                StartCoroutine(WaitForRequest(www));
            }
        }

        IEnumerator WaitForRequest(WWW www)
        {
            Debug.Log("Send request and wait for response");
            yield return www;

            if (www.error == null)
            {
                Debug.Log("WWW OK!: " + www.text);
                digitResult.text = www.text;
            }
            else
            {
                Debug.Log("WWW Error: " + www.error);
            }
        }

        public void StartDetectFigure()
        {
            m_Flip = !m_Flip;
            m_DetectFigure = true;
            m_DetectCard = false;
            m_DetectSquare = false;
            m_DetectDigit = false;
        }

        public void StopDetectFigure()
        {
            m_DetectFigure = false;
        }

        public void StartDetectCard()
        {
            m_DetectCard = true;
            m_DetectFigure = false;
            m_DetectDigit = false;
            m_DetectSquare = false;
        }

        public void StopDetectCard()
        {
            m_DetectCard = false;
        }

        public void StartDetectDigit()
        {
            m_DetectDigit = true;
        }

        public void StartDetectFig()
        {
            m_DetectFig = true;
        }

    }
}
