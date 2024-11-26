using System;
using System.Collections.Generic;
using System.Linq;
using CNTK;
using System.IO;

namespace Regularization
{
    class Regularization
    {
        const int batchSize = 6;
        const int epochCount = 100000;

        readonly Variable x;
        readonly Function y;

        public Regularization(int hiddenLayerSize)
        {
            int[] layers = new int[] { DataSet.InputSize, hiddenLayerSize, DataSet.OutputSize };

            // Build graph
            x = Variable.InputVariable(new int[] { layers[0] }, DataType.Float);

            Function lastLayer = x;
            for (int i = 0; i < layers.Length - 1; i++)
            {
                Parameter weight = new Parameter(new int[] { layers[i + 1], layers[i] }, DataType.Float, CNTKLib.GlorotNormalInitializer());
                Parameter bias = new Parameter(new int[] { layers[i + 1] }, DataType.Float, CNTKLib.GlorotNormalInitializer());

                Function times = CNTKLib.Times(weight, lastLayer);
                Function plus = CNTKLib.Plus(times, bias);
                lastLayer = CNTKLib.Sigmoid(plus);
            }

            y = lastLayer;
        }

        public void Train(DataSet ds, double l1weight, double l2weight)
        {
            // Extend graph
            Variable yt = Variable.InputVariable(new int[] { DataSet.OutputSize }, DataType.Float);
            Function loss = CNTKLib.SquaredError(y, yt);

            AdditionalLearningOptions alo = new AdditionalLearningOptions();
            alo.l1RegularizationWeight = l1weight;
            alo.l2RegularizationWeight = l2weight;
            Learner learner = CNTKLib.SGDLearner(new ParameterVector(y.Parameters().ToArray()), new TrainingParameterScheduleDouble(10.0, batchSize), alo);
            Trainer trainer = Trainer.CreateTrainer(y, loss, loss, new List<Learner>() { learner });

            // Train
            for (int epochI = 0; epochI <= epochCount; epochI++)
            {
                double sumLoss = 0;

                //ds.Shuffle();
                for (int batchI = 0; batchI < ds.Count / batchSize; batchI++)
                {
                    Value x_value = Value.CreateBatch(x.Shape, ds.Input.GetRange(batchI * batchSize * DataSet.InputSize, batchSize * DataSet.InputSize), DeviceDescriptor.CPUDevice);
                    Value yt_value = Value.CreateBatch(yt.Shape, ds.Output.GetRange(batchI * batchSize * DataSet.OutputSize, batchSize * DataSet.OutputSize), DeviceDescriptor.CPUDevice);
                    var inputDataMap = new Dictionary<Variable, Value>()
                    {
                        { x, x_value },
                        { yt, yt_value }
                    };

                    trainer.TrainMinibatch(inputDataMap, false, DeviceDescriptor.CPUDevice);
                    sumLoss += trainer.PreviousMinibatchLossAverage() * trainer.PreviousMinibatchSampleCount();
                }
            }
        }

        public void Evaluate(DataSet ds, out double lossValue)
        {
            // Extend graph
            Variable yt = Variable.InputVariable(new int[] { DataSet.OutputSize }, DataType.Float);
            Function loss = CNTKLib.SquaredError(y, yt);

            Evaluator evaluator_loss = CNTKLib.CreateEvaluator(loss);

            double sumLoss = 0;
            for (int batchI = 0; batchI < ds.Count / batchSize; batchI++)
            {
                Value x_value = Value.CreateBatch(x.Shape, ds.Input.GetRange(batchI * batchSize * DataSet.InputSize, batchSize * DataSet.InputSize), DeviceDescriptor.CPUDevice);
                Value yt_value = Value.CreateBatch(yt.Shape, ds.Output.GetRange(batchI * batchSize * DataSet.OutputSize, batchSize * DataSet.OutputSize), DeviceDescriptor.CPUDevice);
                var inputDataMap = new UnorderedMapVariableValuePtr()
                    {
                        { x, x_value },
                        { yt, yt_value }
                    };

                sumLoss += evaluator_loss.TestMinibatch(inputDataMap, DeviceDescriptor.CPUDevice) * batchSize;
            }
            lossValue = sumLoss / ds.Count;
        }

        public float Prediction(float value)
        {
            Value x_value = Value.CreateBatch(x.Shape, new float[] { value }, DeviceDescriptor.CPUDevice);
            var inputDataMap = new Dictionary<Variable, Value>() { { x, x_value } };
            var outputDataMap = new Dictionary<Variable, Value>() { { y, null } };
            y.Evaluate(inputDataMap, outputDataMap, DeviceDescriptor.CPUDevice);

            return DataSet.DeNormalizeResult(new List<float>() { outputDataMap[y].GetDenseData<float>(y)[0][0] })[0];
        }

        public void Map(String filename, float from, float to, float increment)
        {
            using (System.IO.StreamWriter file = new System.IO.StreamWriter(filename))
            {
                for (float value = from; value <= to; value += increment)
                {
                    float prediction = Prediction(value);
                    file.WriteLine(String.Format("{0}\t{1}", value, prediction));
                }
            }
        }
    }

    public class DataSet
    {
        public const int InputSize = 1;
        public List<float> Input { get; set; } = new List<float>();

        public const int OutputSize = 1;
        public List<float> Output { get; set; } = new List<float>();

        public int Count { get; set; }

        public DataSet(string filename)
        {
            LoadData(filename);
        }

        void LoadData(string filename)
        {
            Count = 0;
            foreach (String line in File.ReadAllLines(filename))
            {
                var floats = Normalize(line.Split('\t').Select(x => float.Parse(x)).ToList());
                Input.AddRange(floats.GetRange(0, InputSize));
                Output.Add(floats[InputSize]);
                Count++;
            }
        }

        public void Shuffle()
        {
            Random rnd = new Random();
            for (int swapI = 0; swapI < Count; swapI++)
            {
                var a = rnd.Next(Count);
                var b = rnd.Next(Count);
                if (a != b)
                {
                    float T;
                    for (int i = 0; i < InputSize; i++)
                    {
                        T = Input[a * InputSize + i];
                        Input[a * InputSize + i] = Input[b * InputSize + i];
                        Input[b * InputSize + i] = T;
                    }
                    T = Output[a]; Output[a] = Output[b]; Output[b] = T;
                }
            }
        }

        static float[] minValues;
        static float[] maxValues;

        public static List<float> Normalize(List<float> floats)
        {
            List<float> normalized = new List<float>();
            for (int i = 0; i < floats.Count; i++)
                normalized.Add((floats[i] - minValues[i]) / (maxValues[i] - minValues[i]));
            return normalized;
        }

        public static List<float> DeNormalizeResult(List<float> floats)
        {
            List<float> denormalized = new List<float>();
            for (int i = 0; i < floats.Count; i++)
                denormalized.Add(floats[i] * (maxValues[i + InputSize] - minValues[i + InputSize]) + minValues[i + InputSize]);
            return denormalized;
        }

        public static void LoadMinMax(string filename)
        {
            foreach (String line in File.ReadAllLines(filename))
            {
                var floats = line.Split('\t').Select(x => float.Parse(x)).ToList();
                if (minValues == null)
                {
                    minValues = floats.ToArray();
                    maxValues = floats.ToArray();
                }
                else
                {
                    for (int i = 0; i < floats.Count; i++)
                        if (floats[i] < minValues[i])
                            minValues[i] = floats[i];
                        else
                            if (floats[i] > maxValues[i])
                            maxValues[i] = floats[i];
                }
            }
        }
    }

    public class Program
    {
        DataSet testDS;
        DataSet trainSmallCleanDS;
        DataSet trainSmallNoisyDS;
        DataSet trainLargeNoisyDS;

        void RunCase(string caseID, DataSet trainDS, int hiddenLayerSize, double l1weight, double l2weight)
        {
            Regularization app = new Regularization(hiddenLayerSize);
            app.Train(trainDS, l1weight, l2weight);
            double trainLoss, testLoss;
            app.Evaluate(trainDS, out trainLoss);
            app.Evaluate(testDS, out testLoss);
            app.Map(@"..\data\regularization\Result_" + caseID + ".txt", 0.0f, 1.0f, 0.01f);
            Console.WriteLine(String.Format("{0}\t{1:e}\t{2:e}", caseID, trainLoss, testLoss));
        }

        void RunAllCases()
        {
            DataSet.LoadMinMax(@"..\data\regularization\train-large-noisy.txt");

            testDS = new DataSet(@"..\data\regularization\test.txt");
            trainSmallCleanDS = new DataSet(@"..\data\regularization\train-small-clean.txt");
            trainSmallNoisyDS = new DataSet(@"..\data\regularization\train-small-noisy.txt");
            trainLargeNoisyDS = new DataSet(@"..\data\regularization\train-large-noisy.txt");
            RunCase("sc1", trainSmallCleanDS, 1, 0, 0);
            RunCase("sc5", trainSmallCleanDS, 5, 0, 0);
            RunCase("sc10", trainSmallCleanDS, 10, 0, 0);
            RunCase("sn1", trainSmallNoisyDS, 1, 0, 0);
            RunCase("sn5", trainSmallNoisyDS, 5, 0, 0);
            RunCase("sn10", trainSmallNoisyDS, 10, 0, 0);
            RunCase("ln1", trainLargeNoisyDS, 1, 0, 0);
            RunCase("ln5", trainLargeNoisyDS, 5, 0, 0);
            RunCase("ln10", trainLargeNoisyDS, 10, 0, 0);
            RunCase("sn5_l1-5", trainSmallNoisyDS, 5, 0.00001, 0);
            RunCase("sn5_l1-4", trainSmallNoisyDS, 5, 0.0001, 0);
            RunCase("sn5_l1-3", trainSmallNoisyDS, 5, 0.001, 0);
            RunCase("sn5_l1-2", trainSmallNoisyDS, 5, 0.01, 0);
            RunCase("sn5_l2-5", trainSmallNoisyDS, 5, 0, 0.00001);
            RunCase("sn5_l2-4", trainSmallNoisyDS, 5, 0, 0.0001);
            RunCase("sn5_l2-3", trainSmallNoisyDS, 5, 0, 0.001);
            RunCase("sn5_l2-2", trainSmallNoisyDS, 5, 0, 0.01);
        }

        static void Main(string[] args)
        {
            new Program().RunAllCases();
        }
    }
}
