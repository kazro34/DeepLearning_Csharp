using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using CNTK;

namespace Linear_Regression
{
    class LinearFunctionApproximation
    {
        string[] trainData = File.ReadAllLines(@"..\data\LinearfunctionApproximation.txt");
        Random rnd = new Random();

        void RunGradientDescentMatrix()
        {
            int n =trainData.Length;

            //Build graph
            Variable x = Variable.InputVariable(new int[] { n, 3 }, DataType.Float);
            Variable yt = Variable.InputVariable(new int[] { n, 1}, DataType.Float);
            Parameter w = new Parameter(new int[] { 3, 1 }, DataType.Float, rnd.NextDouble());
            Function y = CNTKLib.Times(x, w);

            Function sqDiff = CNTKLib.Square(CNTKLib.Minus(y, yt));
            Function loss = CNTKLib.ReduceSum(sqDiff, Axis.AllAxes());
            Learner learner = CNTKLib.SGDLearner(new ParameterVector() { w }, new TrainingParameterScheduleDouble(0.001, 1));
            Trainer trainer = Trainer.CreateTrainer(loss, loss, null, new List<Learner>() { learner });

            //Prepare data
            float[] x_data = new float[n*3];
            float[] yt_data = new float[n];
            for (int i = 0; i < n; i++) 
            {
                var floats = trainData[i].Split('\t').Select(xx => float.Parse(xx)).ToList();
                x_data[i] = floats[0];
                x_data[n + i]= floats[1];
                x_data[n * 2 + i] = 1.0f;
                yt_data[i] = floats[2];
            }
            Value x_value = Value.CreateBatch(new int[] { n, 3 }, x_data, DeviceDescriptor.CPUDevice);
            Value yt_value = Value.CreateBatch(new int[] { n, 1 }, yt_data, DeviceDescriptor.CPUDevice);

            var inputDataMap = new Dictionary<Variable, Value>()
                    {
                        { x, x_value },
                        { yt, yt_value }
                    };

            // Train
            for (int i = 0; i < 500; i++)
            {
                trainer.TrainMinibatch(inputDataMap, true, DeviceDescriptor.CPUDevice);
                float w1 = new Value(w.GetValue()).GetDenseData<float>(w)[0][0];
                float w2 = new Value(w.GetValue()).GetDenseData<float>(w)[0][1];
                float w3 = new Value(w.GetValue()).GetDenseData<float>(w)[0][2];
                Console.WriteLine(String.Format("{0}\ta:{1}\tb:{2}\tc:{3}", i, w1, w2, w3));
            }


        }
        void RunGradientDescentVector()
        {
            //Build graph
            //x * w = yt
            Variable x = Variable.InputVariable(new int[] { 1, 3 }, DataType.Float);
            Variable yt = Variable.InputVariable(new int[] { 1 }, DataType.Float);
            Parameter w = new Parameter(new int[] { 3, 1 }, DataType.Float, rnd.NextDouble());
            Function y = CNTKLib.Times(x, w);

            Function loss = CNTKLib.Square(CNTKLib.Minus(y, yt));
            Learner learner = CNTKLib.SGDLearner(new ParameterVector() { w }, new TrainingParameterScheduleDouble(0.001, 1));
            Trainer trainer = Trainer.CreateTrainer(loss, loss, null, new List<Learner>() { learner });

            // Train


            for (int i = 0; i < 500; i++)
            {
                foreach (string line in trainData)
                {
                    var floats = line.Split('\t').Select(xx => float.Parse(xx)).ToList();
                    Value x_value = Value.CreateBatch(new int[] { 1, 3 }, new float[] { floats[0], floats[1], 1.0f }, DeviceDescriptor.CPUDevice);
                    Value yt_value = Value.CreateBatch(new int[] { 1 }, new float[] { floats[2] }, DeviceDescriptor.CPUDevice);
                    var inputDataMap = new Dictionary<Variable, Value>()
                    {
                        { x, x_value },
                        { yt, yt_value }
                    };
                    trainer.TrainMinibatch(inputDataMap, true, DeviceDescriptor.CPUDevice);
                }
                float w1 = new Value(w.GetValue()).GetDenseData<float>(w)[0][0];
                float w2 = new Value(w.GetValue()).GetDenseData<float>(w)[0][1];
                float w3 = new Value(w.GetValue()).GetDenseData<float>(w)[0][2];
                Console.WriteLine(String.Format("{0}\ta:{1}\tb:{2}\tc:{3}", i, w1, w2, w3));
            }

        }
        void RunGradientDescentScalar()
        {
            //Build graph
            //a*x + b*y = yt
            Variable x1 = Variable.InputVariable(new int[] { 1 }, DataType.Float);
            Variable x2 = Variable.InputVariable(new int[] { 1 }, DataType.Float);
            Variable yt = Variable.InputVariable(new int[] { 1 }, DataType.Float);
            Parameter a = new Parameter(new int[] { 1 }, DataType.Float, rnd.NextDouble());
            Parameter b = new Parameter(new int[] { 1 }, DataType.Float, rnd.NextDouble());
            Parameter c = new Parameter(new int[] { 1 }, DataType.Float, rnd.NextDouble());
            Function y = CNTKLib.Plus(CNTKLib.Plus(CNTKLib.Times(a, x1), CNTKLib.Times(b, x2)), c);

            Function loss = CNTKLib.Square(CNTKLib.Minus(y, yt));
            Learner learner = CNTKLib.SGDLearner(new ParameterVector() { a, b, c }, new TrainingParameterScheduleDouble(0.001, 1));
            Trainer trainer = Trainer.CreateTrainer(loss, loss, null, new List<Learner>() { learner });

            // Train

            for (int i = 0; i < 500; i++)
            {
                foreach (string line in trainData)
                {
                    var floats = line.Split('\t').Select(xx => float.Parse(xx)).ToList();
                    Value x1_value = Value.CreateBatch(new int[] { 1 }, new float[] { floats[0] }, DeviceDescriptor.CPUDevice);
                    Value x2_value = Value.CreateBatch(new int[] { 1 }, new float[] { floats[1] }, DeviceDescriptor.CPUDevice);
                    Value yt_value = Value.CreateBatch(new int[] { 1 }, new float[] { floats[2] }, DeviceDescriptor.CPUDevice);
                    var inputDataMap = new Dictionary<Variable, Value>()
                    {
                        { x1, x1_value },
                        { x2, x2_value },
                        { yt, yt_value }
                    };
                    trainer.TrainMinibatch(inputDataMap, true, DeviceDescriptor.CPUDevice);
                }
                float aValue = new Value(a.GetValue()).GetDenseData<float>(a)[0][0];
                float bValue = new Value(b.GetValue()).GetDenseData<float>(b)[0][0];
                float cValue = new Value(c.GetValue()).GetDenseData<float>(c)[0][0];
                Console.WriteLine(  String.Format("{0}\ta:{1}\tb:{2}\tc:{3}", i, aValue, bValue, cValue));
            }
        }
        static void Main(string[] args)
        {
            //new LinearFunctionApproximation().RunGradientDescentScalar();
            //new LinearFunctionApproximation().RunGradientDescentVector();
            new LinearFunctionApproximation().RunGradientDescentMatrix();

        }

    }
}