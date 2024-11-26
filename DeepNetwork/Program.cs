using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using CNTK;

namespace DeepNetwork
{
    class DeepNetwork
    {
        readonly int[] layers = new int[] { DataSet.InputSize, 10, 10, 10, DataSet.OutputSize };

        const int batchSize = 50;
        const int epochCount = 100;

        readonly Variable x;
        readonly Function y;

        public DeepNetwork()
        {
            //Build graph
            x = Variable.InputVariable(new int[] { layers[0] }, DataType.Float, "x");

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
        public DeepNetwork(string filename)
        {
            // Load graph
            y = Function.Load(filename, DeviceDescriptor.CPUDevice);
            x = y.Arguments.First(x => x.Name == "x");
        }

        public void Train(DataSet ds)
        {
            //Extend graph
            Variable yt = Variable.InputVariable(new int[] { DataSet.OutputSize }, DataType.Float);
            Function loss = CNTKLib.BinaryCrossEntropy(y, yt);

            Function y_rounded = CNTKLib.Round(y);
            Function y_yt_equal = CNTKLib.Equal(y_rounded, yt);

            Learner learner = CNTKLib.SGDLearner(new ParameterVector(y.Parameters().ToArray()), new TrainingParameterScheduleDouble(0.01, 1));
            Trainer trainer = Trainer.CreateTrainer(y, loss, y_yt_equal, new List<Learner>() { learner });

            //Train
            for (int epochI = 0; epochI <= epochCount; epochI++)
            {
                double sumLoss = 0;
                double sumEval = 0;


                for (int batchI = 0; batchI < ds.Count / batchSize; batchI++)
                {
                    Value x_value = Value.CreateBatch(x.Shape, ds.Input.GetRange(batchI * batchSize * DataSet.InputSize, batchSize * DataSet.InputSize), DeviceDescriptor.CPUDevice);
                    Value yt_value = Value.CreateBatch(yt.Shape, ds.Output.GetRange(batchI * batchSize * DataSet.OutputSize, batchSize * DataSet.OutputSize), DeviceDescriptor.CPUDevice);
                    var inputDataMap = new Dictionary<Variable, Value>()
                    {
                        { x, x_value },
                        { yt,  yt_value }
                    };

                    trainer.TrainMinibatch(inputDataMap, false, DeviceDescriptor.CPUDevice);
                    sumLoss += trainer.PreviousMinibatchLossAverage() * trainer.PreviousMinibatchSampleCount();
                    sumEval += trainer.PreviousMinibatchEvaluationAverage() * trainer.PreviousMinibatchSampleCount();
                }
                Console.WriteLine(String.Format("{0}\tloss:{1}\teval:{2}", epochI, sumLoss / ds.Count, sumEval / ds.Count));
            }
        }
        public void Save(string filename)
        {
            y.Save(filename);
        }
        public double Evaluate(DataSet ds)
        {
            Variable yt = Variable.InputVariable(new int[] { DataSet.OutputSize }, DataType.Float);

            Function y_rounded = CNTKLib.Round(y);
            Function y_yt_equal = CNTKLib.Equal(y_rounded, yt);
            Evaluator evaluator = CNTKLib.CreateEvaluator(y_yt_equal);

            double sumEval = 0;
            for (int batchI = 0; batchI < ds.Count; batchI++)
            {
                Value x_value = Value.CreateBatch(x.Shape, ds.Input.GetRange(batchI * batchSize * DataSet.InputSize, batchSize * DataSet.InputSize), DeviceDescriptor.CPUDevice);
                Value yt_value = Value.CreateBatch(yt.Shape, ds.Output.GetRange(batchI * batchSize * DataSet.OutputSize, batchSize * DataSet.OutputSize), DeviceDescriptor.CPUDevice);
                var inputDataMap = new UnorderedMapVariableValuePtr()
                    {
                        { x, x_value },
                        { yt,  yt_value }
                    };

                sumEval += evaluator.TestMinibatch(inputDataMap, DeviceDescriptor.CPUDevice) * batchSize;
            }
            return sumEval / ds.Count;

        }

    }
    public class DataSet
    {
        public const int InputSize = 4;
        public List<float> Input { get; set; } = new List<float>();

        public const int OutputSize = 1;
        public List<float> Output { get; set; } = new List<float>();

        public int Count { get; set; }
        public DataSet(string filename)
        {
            LoadData(filename);
        }
        public void LoadData(string filename)
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
                normalized.Add(floats[i] - minValues[i] / (maxValues[i] - minValues[i]));
            return normalized;
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
                    {
                        if (floats[i] < minValues[i])
                            minValues[i] = floats[i];
                        else
                            if (floats[i] > maxValues[i])
                            maxValues[i] = floats[i];
                    }
                }
            }
        }
    }

    public class Program
    {
        static void Main(string[] args)
        {
            DataSet.LoadMinMax(@"..\data\HusbandEvaluation.txt");
            DataSet trainDS = new DataSet(@"..\data\HusbandEvaluation.txt");
            DataSet testDS = new DataSet(@"..\data\HusbandEvaluation-test.txt");


            DeepNetwork app = new DeepNetwork();
            app.Train(trainDS);
            Console.WriteLine("Eval train:" + app.Evaluate(trainDS));
            Console.WriteLine("Eval test:" + app.Evaluate(testDS));
        }
    }

}
