using UnityEngine;
using System.Collections;
using System.Collections.Generic;

using OpenCVForUnity;
using System.IO;
using System;

namespace OpenCVForUnitySample
{
    public enum CardType
    {
        Digit = 1,
        Letter = 2
    };

    public class Card
    {
        public CvKNearest m_Knn;
        public static int m_minSquareLen = 10;
        public static int m_maxSquareLen = 400;
        public static double m_maxSquareLenRatio = 2;
        public long m_distanceThreshold;

        public int m_sampleSize;

        List<Point> m_StdSquareClockwise;
        List<Point> m_StdSquareCounterClockwise;

        public Card(CardType cardType)
        {
            m_Knn = new CvKNearest();

            m_sampleSize = 50;
            m_distanceThreshold = 14000000;

            Mat Sample = new Mat(), Response = new Mat();

            if (((int)cardType & (int)CardType.Digit) > 0)
            {
                Sample = ReadData(Application.persistentDataPath + "/cards/50/digit_TrainingData.yml", Sample, true);
                Response = ReadData(Application.persistentDataPath + "/cards/50/digit_LabelData.yml", Response, false);
            }
            if (((int)cardType & (int)CardType.Letter) > 0)
            {
                Sample = ReadData(Application.persistentDataPath + "/cards/50/letter_TrainingData.yml", Sample, true);
                Response = ReadData(Application.persistentDataPath + "/cards/50/letter_LabelData.yml", Response, false);
            }

            m_Knn.train(Sample, Response);

            m_StdSquareClockwise = new List<Point>(4);
            m_StdSquareCounterClockwise = new List<Point>(4);
            m_StdSquareClockwise.Add(new Point(0, 0));
            m_StdSquareClockwise.Add(new Point(m_sampleSize, 0));
            m_StdSquareClockwise.Add(new Point(m_sampleSize, m_sampleSize));
            m_StdSquareClockwise.Add(new Point(0, m_sampleSize));
            m_StdSquareCounterClockwise.Add(new Point(0, 0));
            m_StdSquareCounterClockwise.Add(new Point(0, m_sampleSize));
            m_StdSquareCounterClockwise.Add(new Point(m_sampleSize, m_sampleSize));
            m_StdSquareCounterClockwise.Add(new Point(m_sampleSize, 0));

        }

        public static Mat ReadData(string fileName, Mat mat, bool verticalCombine)
        {
            String[] lines = File.ReadAllLines(fileName);
            String line, dataStr;
            int rows = -1;
            int cols = -1;
            int lineIndex;
            bool importData = false;
            for (lineIndex = 0; lineIndex < lines.Length; lineIndex++)
            {
                line = lines[lineIndex].Trim();
                if (line.StartsWith("rows"))
                    rows = int.Parse(line.Split(':')[1].Trim());
                if (line.StartsWith("cols"))
                    cols = int.Parse(line.Split(':')[1].Trim());
                if (line.StartsWith("data:") && rows > 0 && cols > 0)
                {
                    importData = true;
                    break;
                }
            }
            if (importData == false)
                return mat;
            float[] data = new float[rows * cols];
            Mat result = new Mat(rows, cols, CvType.CV_32FC1);
            int dataIndex = 0;
            for (int i = lineIndex; i < lines.Length; i++)
            {
                dataStr = lines[i].IndexOf('[') == -1 ? lines[i] : lines[i].Split('[')[1];
                dataStr = dataStr.IndexOf(']') == -1 ? dataStr : dataStr.Split(']')[0];
                String[] lineDataAry = dataStr.Split(',');
                for (int j = 0; j < lineDataAry.Length; j++)
                {
                    if (lineDataAry[j].Trim() == "")
                        continue;
                    data[dataIndex] = float.Parse(lineDataAry[j].Trim());
                    dataIndex++;
                }
            }
            result.put(0, 0, data);

            if (mat.rows() == 0)
                return result;

            Mat combine = new Mat(rows + mat.rows(), cols + mat.cols(), CvType.CV_32FC1);
            List<Mat> matList = new List<Mat>(2);
            matList.Add(mat);
            matList.Add(result);
            if (verticalCombine)
            {
                Core.vconcat(matList, combine);
            }
            else
            {
                Core.hconcat(matList, combine);
            }

            return combine;
        }

        public static List<List<Point>> FilterSquares(List<List<Point>> squares)
        {
            List<List<Point>> filterSquares = new List<List<Point>>();

            for (int j = 0; j < squares.Count; j++)
            {
                if (FigProc.IsSquareClockwise(squares[j]))
                {
                    continue;
                }
                double len, curMaxLen = 0, curMinLen = 10000;

                for (int i = 0; i < 3; i++)
                {
                    len = FigProc.FindLen(squares[j][i % 4], squares[j][(i + 1) % 4]);
                    curMaxLen = len > curMaxLen ? len : curMaxLen;
                    curMinLen = len < curMinLen ? len : curMinLen;
                }

                if (curMaxLen > m_maxSquareLen || curMinLen < m_minSquareLen || curMaxLen / curMinLen > m_maxSquareLenRatio)
                {
                    continue;
                }
                filterSquares.Add(squares[j]);
            }
            return filterSquares;
        }

        public List<List<Point>> Recgonize(Mat image, out List<int> resultVec, bool one = false)
        {
            resultVec = new List<int>();

            List<List<Point>> squares = new List<List<Point>>();
            List<List<Point>> filterSquares;

            squares = FigProc.FindSquares(image.clone());
            filterSquares = Card.FilterSquares(squares);

            Mat transform = new Mat();
            Mat transResult = new Mat();

            Mat result = new Mat();
            Mat distance = new Mat();

            double minDist = -1;
            int targetValue = -1;
            for (int i = 0; i < filterSquares.Count; i++)
            {
                bool clockwise = FigProc.IsSquareClockwise(filterSquares[i]);
                transform = Calib3d.findHomography(new MatOfPoint2f(filterSquares[i].ToArray()), clockwise ? new MatOfPoint2f(m_StdSquareClockwise.ToArray()) : new MatOfPoint2f(m_StdSquareClockwise.ToArray()));

                Imgproc.warpPerspective(image, transResult, transform, new Size(image.cols(), image.rows()));
                OpenCVForUnity.Rect rect = new OpenCVForUnity.Rect(0, 0, m_sampleSize, m_sampleSize);
                Mat tgt = transResult.submat(rect);
                tgt.convertTo(tgt, CvType.CV_32FC1);
                m_Knn.find_nearest(tgt.reshape(1, 1), 1, new Mat(), result, distance);
                if (distance.get(0, 0)[0] < m_distanceThreshold)
                {
                    if (one)
                    {
                        if (minDist < 0 || distance.get(0, 0)[0] < minDist)
                        {
                            minDist = distance.get(0, 0)[0];
                            targetValue = (int)result.get(0, 0)[0] / 4;
                        }
                    }
                    else
                    {
                        int value = (int)result.get(0, 0)[0] / 4;
                        if (resultVec.Contains(value) == false)
                        {
                            resultVec.Add(value);
                        }
                    }

                }
            }
            if (one && targetValue >= 0)
                resultVec.Add(targetValue);
            return filterSquares;
        }
    }
}