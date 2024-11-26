using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using CNTK;
using MNIST;

namespace MNISTClassification
{
    class MNISTClassification
    {
        const int batchSize = 1000;
        const int epochCount = 100;

        readonly Variable x;
        readonly Function y;

        public MNISTClassification(int hiddenNeuronCount)
        {
            int[] layers = new int[] { MNISTDataSet.InputSize, hiddenNeuronCount, hiddenNeuronCount, MNISTDataSet.OutputSize };

            // Build graph
            x = Variable.InputVariable(new int[] { layers[0] }, DataType.Float);

            Function lastLayer = x;
            for (int i = 0; i < layers.Length - 1; i++)
            {
                Parameter weight = new Parameter(new int[] { layers[i + 1], layers[i] }, DataType.Float, CNTKLib.GlorotNormalInitializer());
                Parameter bias = new Parameter(new int[] { layers[i + 1] }, DataType.Float, CNTKLib.GlorotNormalInitializer());

                Function times = CNTKLib.Times(weight, lastLayer);
                Function plus = CNTKLib.Plus(times, bias);

                if (i != layers.Length - 2)
                    lastLayer = CNTKLib.Sigmoid(plus);
                else
                    lastLayer = CNTKLib.Softmax(plus);
            }

            y = lastLayer;
        }

        public void Train(MNISTDataSet ds)
        {
            // Extend graph
            Variable yt = Variable.InputVariable(new int[] { MNISTDataSet.OutputSize }, DataType.Float);
            Function loss = CNTKLib.CrossEntropyWithSoftmax(y, yt);
            Function err = CNTKLib.ClassificationError(y, yt);

            Learner learner = CNTKLib.SGDLearner(new ParameterVector(y.Parameters().ToArray()), new TrainingParameterScheduleDouble(1.0, batchSize));
            Trainer trainer = Trainer.CreateTrainer(y, loss, err, new List<Learner>() { learner });

            // Train
            for (int epochI = 0; epochI <= epochCount; epochI++)
            {
                double sumLoss = 0;
                double sumError = 0;

                //ds.Shuffle();
                for (int batchI = 0; batchI < ds.Count / batchSize; batchI++)
                {
                    Value x_value = Value.CreateBatch(x.Shape, ds.Input.GetRange(batchI * batchSize * MNISTDataSet.InputSize, batchSize * MNISTDataSet.InputSize), DeviceDescriptor.CPUDevice);
                    Value yt_value = Value.CreateBatch(yt.Shape, ds.Output.GetRange(batchI * batchSize * MNISTDataSet.OutputSize, batchSize * MNISTDataSet.OutputSize), DeviceDescriptor.CPUDevice);
                    var inputDataMap = new Dictionary<Variable, Value>()
                    {
                        { x, x_value },
                        { yt, yt_value }
                    };

                    trainer.TrainMinibatch(inputDataMap, false, DeviceDescriptor.CPUDevice);
                    sumLoss += trainer.PreviousMinibatchLossAverage() * trainer.PreviousMinibatchSampleCount();
                    sumError += trainer.PreviousMinibatchEvaluationAverage() * trainer.PreviousMinibatchSampleCount();
                }
                Console.WriteLine(String.Format("{0}\t{1:0.0000}\t{2:0.0000}", epochI, sumLoss / ds.Count, 1.0 - sumError / ds.Count));
            }
        }

        public void Evaluate(MNISTDataSet ds, out double lossValue, out double accValue)
        {
            // Extend graph
            Variable yt = Variable.InputVariable(new int[] { MNISTDataSet.OutputSize }, DataType.Float);
            Function loss = CNTKLib.CrossEntropyWithSoftmax(y, yt);
            Function err = CNTKLib.ClassificationError(y, yt);

            Evaluator evaluator_loss = CNTKLib.CreateEvaluator(loss);
            Evaluator evaluator_err = CNTKLib.CreateEvaluator(err);

            double sumEval = 0;
            double sumLoss = 0;
            for (int batchI = 0; batchI < ds.Count / batchSize; batchI++)
            {
                Value x_value = Value.CreateBatch(x.Shape, ds.Input.GetRange(batchI * batchSize * MNISTDataSet.InputSize, batchSize * MNISTDataSet.InputSize), DeviceDescriptor.CPUDevice);
                Value yt_value = Value.CreateBatch(yt.Shape, ds.Output.GetRange(batchI * batchSize * MNISTDataSet.OutputSize, batchSize * MNISTDataSet.OutputSize), DeviceDescriptor.CPUDevice);
                var inputDataMap = new UnorderedMapVariableValuePtr()
                    {
                        { x, x_value },
                        { yt, yt_value }
                    };

                sumLoss += evaluator_loss.TestMinibatch(inputDataMap) * batchSize;
                sumEval += evaluator_err.TestMinibatch(inputDataMap) * batchSize;
            }
            lossValue = sumLoss / ds.Count;
            accValue = 1 - sumEval / ds.Count;
        }

        public String VisibleTest(MNISTDataSet ds, int count)
        {
            Value x_value = Value.CreateBatch(x.Shape, ds.Input.GetRange(0, count * MNISTDataSet.InputSize), DeviceDescriptor.CPUDevice);
            var inputDataMap = new UnorderedMapVariableValuePtr()
                    {
                        { x, x_value }
                    };
            var outputDataMap = new UnorderedMapVariableValuePtr()
                    {
                        { y, null }
                    };

            y.Evaluate(inputDataMap, outputDataMap, DeviceDescriptor.CPUDevice);
            IList<IList<float>> resultValue = outputDataMap[y].GetDenseData<float>(y);

            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < count; i++)
            {
                sb.Append(ds.DataToString(i));
                sb.Append("\nResult:[");
                int max = 0;
                for (int d = 0; d <= 9; d++)
                {
                    if (resultValue[i][d] > resultValue[i][max])
                        max = d;
                    if (d != 0) sb.Append(", ");
                    sb.Append(String.Format("{0:0.00}", resultValue[i][d]));
                }
                sb.Append("]\nPrediction:").Append(max).Append("\n\n");
            }
            return sb.ToString();
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            int hiddenNeuronCount = 10;

            MNISTDataSet trainDS = new MNISTDataSet(@"..\data\mnist\train-labels.idx1-ubyte", @"..\data\mnist\train-images.idx3-ubyte");
            MNISTDataSet testDS = new MNISTDataSet(@"..\data\mnist\t10k-labels.idx1-ubyte", @"..\data\mnist\t10k-images.idx3-ubyte");
            //for (int i = 0; i <= 10; i++)
            //    Console.WriteLine(trainDS.DataToString(i));

            MNISTClassification app = new MNISTClassification(hiddenNeuronCount);
            app.Train(trainDS);
            app.Evaluate(trainDS, out double trainLoss, out double trainAcc);
            app.Evaluate(testDS, out double testLoss, out double testAcc);
            Console.WriteLine(String.Format("Final evaluation:{0}\t{1:0.0000}\t{2:0.0000}\t{3:0.0000}\t{4:0.0000}", hiddenNeuronCount, trainLoss, trainAcc, testLoss, testAcc));
            Console.WriteLine(app.VisibleTest(trainDS, 10));
        }
    }
}
