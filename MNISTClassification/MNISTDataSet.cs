using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace MNIST
{
    public class MNISTDataSet
    {
        int width, height;
        public const int InputSize = 28 * 28;
        public List<float> Input { get; set; } = new List<float>();

        public const int OutputSize = 10;
        public List<float> Output { get; set; } = new List<float>();

        public int Count { get; set; }
        public MNISTDataSet(string indexFilename, string imageFilename)
        {
            LoadIndex(indexFilename);
            LoadImage(imageFilename);
        }
        void LoadIndex(string filename)
        {
            byte[] data = File.ReadAllBytes(filename);
            Count = BitConverter.ToInt32(new byte[] { data[7], data[6], data[5], data[4] }, 0);
            for (int i = 0; i < Count; i++)
                for (int d = 0; d < 9; d++)
                {
                    Output.Add(d == data[8 + i] ? 1.0f : 0.0f);
                }
        }
        void LoadImage(string filename)
        {
            byte[] data = File.ReadAllBytes(filename);
            Count = BitConverter.ToInt32(new byte[] { data[7], data[6], data[5], data[4] }, 0);
            width = BitConverter.ToInt32(new byte[] { data[11], data[10], data[9], data[8] }, 0);
            height = BitConverter.ToInt32(new byte[] { data[15], data[14], data[13], data[12] }, 0);
            for (int i = 0; i < Count * InputSize; i++)
                Input.Add(data[16 + i]);
        }
        public String DataToString(int index)
        {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < InputSize; i++)
            {
                if (i % width == 0)
                {
                    if (i/width == height /2)
                    {
                        sb.Append("\t [");
                        for (int d = 0; d <= 9; d++)
                        {
                            if (d != 0) sb.Append(", ");
                            sb.Append(String.Format("{0:0.00}", Output[index * OutputSize + d]));
                        }
                        sb.Append(']');
                    }
                    sb.Append("\n");
                }
                float data = Input[index * InputSize + i];
                if (data > 200)
                    sb.Append("\u2593");
                else
                if (data > 150)
                    sb.Append("\u2592");
                else
                if (data > 100)
                    sb.Append("\u2591");
                else
                    sb.Append(" ");
            }
            return sb.ToString();
        }
    }
}

