using OpenCvSharp;
using Sdcb.OpenVINO;
using System;
using System.Collections.Generic;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Windows.Forms;
 
namespace C__danbooru_Stable_Diffusion_提示词反推_OpenVINO__Demo
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }
 
        string fileFilter = "*.*|*.bmp;*.jpg;*.jpeg;*.tiff;*.tiff;*.png";
        string image_path = "";
        string model_path;
        Mat image;
 
        StringBuilder sb = new StringBuilder();
        public string[] class_names;
 
        Model rawModel;
        PrePostProcessor pp;
        Model m;
        CompiledModel cm;
        InferRequest ir;
 
        private void button1_Click(object sender, EventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog();
            ofd.Filter = fileFilter;
            if (ofd.ShowDialog() != DialogResult.OK) return;
            pictureBox1.Image = null;
            image_path = ofd.FileName;
            pictureBox1.Image = new Bitmap(image_path);
            textBox1.Text = "";
            image = new Mat(image_path);
        }
 
        private void button2_Click(object sender, EventArgs e)
        {
            if (image_path == "")
            {
                return;
            }
 
            button2.Enabled = false;
            textBox1.Text = "";
            sb.Clear();
            Application.DoEvents();
 
            image = new Mat(image_path);
 
 
            image = new Mat(image_path);
            int w = image.Width;
            int h = image.Height;
 
            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();
 
            float[] input_tensor_data;
 
            image.ConvertTo(image, MatType.CV_32FC3, 1.0 / 255);
            input_tensor_data = Common.ExtractMat(image);
 
            Tensor input_tensor = Tensor.FromArray(input_tensor_data, new Shape(1, 3, h, w));
 
            ir.Inputs[0] = input_tensor;
 
            double preprocessTime = stopwatch.Elapsed.TotalMilliseconds;
            stopwatch.Restart();
 
            ir.Run();
 
            double inferTime = stopwatch.Elapsed.TotalMilliseconds;
 
            stopwatch.Restart();
 
            var result_array = ir.Outputs[0].GetData<float>().ToArray();
 
            double[] scores = new double[result_array.Length];
            for (int i = 0; i < result_array.Length; i++)
            {
                double score = 1 / (1 + Math.Exp(result_array[i] * -1));
                scores[i] = score;
            }
 
            List<ScoreIndex> ltResult = new List<ScoreIndex>();
            ScoreIndex temp;
            for (int i = 0; i < scores.Length; i++)
            {
                temp = new ScoreIndex(i, scores[i]);
                ltResult.Add(temp);
            }
 
            //根据分数倒序排序，取前10个
            var SortedByScore = ltResult.OrderByDescending(p => p.Score).ToList().Take(10);
 
            foreach (var item in SortedByScore)
            {
                sb.Append(class_names[item.Index] + ",");
            }
            sb.Length--; // 将长度减1来移除最后一个字符
 
            sb.AppendLine("");
            sb.AppendLine("------------------");
 
 
            double postprocessTime = stopwatch.Elapsed.TotalMilliseconds;
            stopwatch.Stop();
 
            double totalTime = preprocessTime + inferTime + postprocessTime;
 
            sb.AppendLine($"Preprocess: {preprocessTime:F2}ms");
            sb.AppendLine($"Infer: {inferTime:F2}ms");
            sb.AppendLine($"Postprocess: {postprocessTime:F2}ms");
            sb.AppendLine($"Total: {totalTime:F2}ms");
            textBox1.Text = sb.ToString();
            button2.Enabled = true;
        }
 
        private void Form1_Load(object sender, EventArgs e)
        {
            model_path = "model/ml_danbooru.onnx";
 
            image_path = "test_img/2.jpg";
            pictureBox1.Image = new Bitmap(image_path);
            image = new Mat(image_path);
 
            List<string> str = new List<string>();
            StreamReader sr = new StreamReader("model/lable.txt");
            string line;
            while ((line = sr.ReadLine()) != null)
            {
                str.Add(line);
            }
            class_names = str.ToArray();
 
 
            rawModel = OVCore.Shared.ReadModel(model_path);
            pp = rawModel.CreatePrePostProcessor();
 
            m = pp.BuildModel();
            cm = OVCore.Shared.CompileModel(m, "CPU");
            ir = cm.CreateInferRequest();
 
        }
    }
}