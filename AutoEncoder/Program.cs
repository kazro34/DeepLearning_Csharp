using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using CNTK;
using MNIST;

namespace Autoencoder
{
    class AutoEncoder
    {
        const int hiddenNeuronCount = 30;
        const int batchSize = 1000;
        const int epochCount = 100;

        readonly Variable x, z;
        readonly Function middleLayer;
        readonly Function y, y2;

        public AutoEncoder()
        {
            // Build training graph
            x = Variable.InputVariable(new int[] { MNISTDataSet.InputSize }, DataType.Float);

            Parameter weight1 = new Parameter(new int[] { hiddenNeuronCount, MNISTDataSet.InputSize }, DataType.Float, CNTKLib.GlorotNormalInitializer());
            Parameter bias1 = new Parameter(new int[] { hiddenNeuronCount }, DataType.Float, CNTKLib.GlorotNormalInitializer());
            middleLayer = CNTKLib.Sigmoid(CNTKLib.Plus(CNTKLib.Times(weight1, x), bias1));

            Parameter weight2 = new Parameter(new int[] { MNISTDataSet.InputSize, hiddenNeuronCount }, DataType.Float, CNTKLib.GlorotNormalInitializer());
            Parameter bias2 = new Parameter(new int[] { MNISTDataSet.InputSize }, DataType.Float, CNTKLib.GlorotNormalInitializer());
            y = CNTKLib.Sigmoid(CNTKLib.Plus(CNTKLib.Times(weight2, middleLayer), bias2));

            // Build decoder graph
            z = Variable.InputVariable(new int[] { hiddenNeuronCount }, DataType.Float);
            y2 = CNTKLib.Sigmoid(CNTKLib.Plus(CNTKLib.Times(weight2, z), bias2));
        }

        public void Train(MNISTDataSet ds)
        {
            // Extend graph
            Variable yt = Variable.InputVariable(new int[] { MNISTDataSet.InputSize }, DataType.Float);
            Function loss = CNTKLib.SquaredError(y, yt);

            Learner learner = CNTKLib.SGDLearner(new ParameterVector(y.Parameters().ToArray()), new TrainingParameterScheduleDouble(1.0, batchSize));
            Trainer trainer = Trainer.CreateTrainer(y, loss, loss, new List<Learner>() { learner });

            // Train
            for (int epochI = 0; epochI <= epochCount; epochI++)
            {
                double sumLoss = 0;

                //ds.Shuffle();
                for (int batchI = 0; batchI < ds.Count / batchSize; batchI++)
                {
                    Value x_value = Value.CreateBatch(x.Shape, ds.Input.GetRange(batchI * batchSize * MNISTDataSet.InputSize, batchSize * MNISTDataSet.InputSize), DeviceDescriptor.CPUDevice);
                    var inputDataMap = new Dictionary<Variable, Value>()
                    {
                        { x, x_value },
                        { yt, x_value }
                    };

                    trainer.TrainMinibatch(inputDataMap, false, DeviceDescriptor.CPUDevice);
                    sumLoss += trainer.PreviousMinibatchLossAverage() * trainer.PreviousMinibatchSampleCount();
                }
                Console.WriteLine(String.Format("{0}\t{1:0.0000}", epochI, sumLoss / ds.Count));
            }
        }

        public void Encode(string sourceFilename, string destinationFilename)
        {
            MNISTDataSet ds = new MNISTDataSet(null, sourceFilename);
            List<byte> encodedData = new List<byte>();
            for (int i = 0; i < ds.Count; i++)
            {
                Value x_value = Value.CreateBatch(x.Shape, ds.Input.GetRange(i * MNISTDataSet.InputSize, MNISTDataSet.InputSize), DeviceDescriptor.CPUDevice);
                var inputDataMap = new Dictionary<Variable, Value>()
                    {
                        { x, x_value }
                    };
                var outputDataMap = new Dictionary<Variable, Value>()
                    {
                        { middleLayer, null }
                    };
                middleLayer.Evaluate(inputDataMap, outputDataMap, DeviceDescriptor.CPUDevice);
                IList<IList<float>> result = outputDataMap[middleLayer].GetDenseData<float>(middleLayer);
                foreach (float data in result[0])
                    encodedData.Add((byte)(data * 255));
            }

            File.WriteAllBytes(destinationFilename, encodedData.ToArray());
        }

        public void Decode(string sourceFilename, string destinationFilename)
        {
            byte[] encodedData = File.ReadAllBytes(sourceFilename);
            List<float> floatEncoded = new List<float>();
            foreach (byte b in encodedData)
                floatEncoded.Add(b / 255.0f);

            List<byte> decodedData = new List<byte>();
            decodedData.AddRange(BitConverter.GetBytes((int)28));
            decodedData.AddRange(BitConverter.GetBytes((int)28));
            decodedData.AddRange(BitConverter.GetBytes((int)encodedData.Length / hiddenNeuronCount));
            decodedData.AddRange(BitConverter.GetBytes((int)2051));
            decodedData.Reverse();

            int count = encodedData.Length / hiddenNeuronCount;
            for (int i = 0; i < count; i++)
            {
                Value z_value = Value.CreateBatch(z.Shape, floatEncoded.GetRange(i * hiddenNeuronCount, hiddenNeuronCount), DeviceDescriptor.CPUDevice);
                var inputDataMap = new Dictionary<Variable, Value>()
                    {
                        { z, z_value }
                    };
                var outputDataMap = new Dictionary<Variable, Value>()
                    {
                        { y2, null }
                    };
                y2.Evaluate(inputDataMap, outputDataMap, DeviceDescriptor.CPUDevice);
                IList<IList<float>> result = outputDataMap[y2].GetDenseData<float>(y2);
                foreach (float data in result[0])
                    decodedData.Add((byte)(data * 255));
            }

            File.WriteAllBytes(destinationFilename, decodedData.ToArray());
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            MNISTDataSet trainDS = new MNISTDataSet(@"..\data\mnist\train-labels.idx1-ubyte", @"..\data\mnist\train-images.idx3-ubyte");
            AutoEncoder app = new AutoEncoder();
            app.Train(trainDS);

            app.Encode(@"..\data\mnist\train-images.idx3-ubyte", @"..\data\mnist\encoded.bin");
            app.Decode(@"..\data\mnist\encoded.bin", @"..\data\mnist\decoded-images.idx3-ubyte");

            MNISTDataSet decodedDS = new MNISTDataSet(@"..\data\mnist\train-labels.idx1-ubyte", @"..\data\mnist\decoded-images.idx3-ubyte");
            for (int i = 0; i <= 10; i++)
            {
                Console.WriteLine(trainDS.DataToString(i));
                Console.WriteLine(decodedDS.DataToString(i));
            }
        }
    }
}
