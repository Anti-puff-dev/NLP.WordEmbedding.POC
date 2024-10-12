

using MathUtils;
using System.Globalization;
using System.Text;

string text = "este é um exemplo de texto para demonstrar a construção de uma matriz de embeddings usando pares adjacentes <EOS>";

text = text.ToLower();
text = RemoverAcentos(text);
HashSet<string> vocab = new HashSet<string>(text.Split(' '));
int embeddingDim = 5;
double[][] embeddingMatrix = new double[vocab.Count][];
double[][] embeddingAttention = new double[vocab.Count][];

Random rand = new Random();


for (int i = 0; i < vocab.Count; i++)
{
    embeddingMatrix[i] = new double[embeddingDim];
    for (int j = 0; j < embeddingDim; j++)
    {
        embeddingMatrix[i][j] = rand.NextDouble();
    }
}

Dictionary<(string, string), int> wordPairs = new Dictionary<(string, string), int>();

string[] words = text.Split(' ');
for (int i = 0; i < words.Length - 1; i++)
{
    var pair = (words[i], words[i + 1]);
    if (wordPairs.ContainsKey(pair))
    {
        wordPairs[pair]++;
    }
    else
    {
        wordPairs[pair] = 1;
    }
}

foreach (var pair in wordPairs.Keys)
{
    int indexWord1 = Array.IndexOf(vocab.ToArray(), pair.Item1);
    int indexWord2 = Array.IndexOf(vocab.ToArray(), pair.Item2);
    for (int j = 0; j < embeddingDim; j++)
    {
        embeddingMatrix[indexWord1][j] += embeddingMatrix[indexWord2][j] * wordPairs[pair];
    }
}


for (int i = 0; i < vocab.Count; i++)
{
    double norm = Math.Sqrt(embeddingMatrix[i].Sum(x => x * x));
    for (int j = 0; j < embeddingDim; j++)
    {
        embeddingMatrix[i][j] /= norm;
    }
}


printM(embeddingMatrix);
Console.WriteLine();
Console.WriteLine();
Console.WriteLine();

for (int i = 0; i < vocab.Count; i++)
{
    embeddingMatrix[i] = PositionalEncoding(embeddingMatrix[i], i);
}

printM(embeddingMatrix);
Console.WriteLine();
Console.WriteLine();
Console.WriteLine();


/*double[,] W_Q = new double[3, 3] { { 0.2, 0.35, -0.7 }, { 0.6, -0.17, -0.3 }, { 0.27, -0.48, 0.4 } };
double[,] W_K = new double[3, 3] { { -0.51, -0.13, 0.67 }, { -0.11, 0.29, 0.49 }, { -0.01, 0.78, -0.61 } };
double[,] W_V = new double[3, 3] { { 0.09, -0.07, 0.8 }, { 0.5, 0.12, 0.48 }, { -0.31, 0.7, 0.33 } };*/


double[,] W_Q = new double[,] { { 2, 0, 2 }, { 2, 0, 0 }, { 2, 1, 2 } };
double[,] W_K = new double[,] { { 2, 2, 2 }, { 0, 2, 1 }, { 0, 1, 1 } };
double[,] W_V = new double[,] { { 1, 1, 0 }, { 0, 1, 1 }, { 0, 0, 0 } };


Head[] heads = new Head[vocab.Count];

for (int i = 0; i < vocab.Count; i++)
{
    heads[i] = new Head();
    heads[i].Q = MathX.Multiply(embeddingMatrix[i], W_Q);
    heads[i].K = MathX.Multiply(embeddingMatrix[i], W_K);
    heads[i].V = MathX.Multiply(embeddingMatrix[i], W_V);
}


for (int i= 0; i < vocab.Count; i++)
{
    double[] scores = new double[vocab.Count];

    for (int q = 0; q < vocab.Count; q++)
    {
        scores[q] = MathX.Dot(heads[q].Q, heads[i].K) / Math.Sqrt(embeddingDim);
    }


    List<double[]> attention_heads = new List<double[]>();
    double[] weights = MathX.Softmax(scores);

    for (int q = 0; q < weights.Length; q++)
    {
        attention_heads.Add(MathX.Multiply(weights[q], heads[i].V));
    }

    double[] sum = MathX.Sum(attention_heads);
    embeddingAttention[i] = sum;
}

printM(embeddingAttention);
Console.WriteLine();
Console.WriteLine();
Console.WriteLine();





   Console.WriteLine("Digite uma palavra para encontrar a mais similar:");
   string targetWord = Console.ReadLine().ToLower();
   int targetIndex = Array.IndexOf(vocab.ToArray(), targetWord);

   if (targetIndex == -1)
   {
       Console.WriteLine("Palavra não encontrada no vocabulário.");
       return;
   }

   double maxSimilarity = double.MinValue;
   int mostSimilarIndex = -1;

   for (int i = 0; i < vocab.Count; i++)
   {
       if (i == targetIndex) continue;
       double similarity = 0;
       for (int j = 0; j < embeddingDim; j++)
       {
           similarity += embeddingMatrix[targetIndex][j] * embeddingMatrix[i][j];
       }

       if (similarity > maxSimilarity)
       {
           maxSimilarity = similarity;
           mostSimilarIndex = i;
       }
   }

   string mostSimilarWord = vocab.ElementAt(mostSimilarIndex);
   Console.WriteLine($"A palavra mais similar a '{targetWord}' é '{mostSimilarWord}' com uma similaridade de {maxSimilarity}.");




void print(double[] data)
{
    Console.Write("[");
    foreach (double x in data)
    {
        Console.Write(x + " ");
    }
    Console.Write("]");
    Console.WriteLine();
}


void printM(double[][] data)
{
    Console.Write("[");
    for (int i = 0; i < data.Length; i++)
    {
        Console.Write("[");
        for (int j = 0; j < data[i].Length; j++)
        {
            Console.Write(data[i][j] + " ");
        }
        Console.Write("]");
    }
    Console.Write("]");
    Console.WriteLine();
}


string RemoverAcentos(string input) 
{
    string normalizedString = input.Normalize(NormalizationForm.FormD);
    StringBuilder builder = new StringBuilder();
    foreach (char c in normalizedString)
    {
        if (CharUnicodeInfo.GetUnicodeCategory(c) != UnicodeCategory.NonSpacingMark)
        {
            builder.Append(c);
        }
    }
    return builder.ToString();
}



double[] PositionalEncoding(double[] wordEmbedding, int position)
{
    int embeddingSize = wordEmbedding.Length;

    double[] positionalEncoding = new double[embeddingSize];
    double angle;

    for (int i = 0; i < embeddingSize; i++)
    {
        if (i % 2 == 0)
        {
            angle = position / Math.Pow(10000, (double)i / embeddingSize);
            positionalEncoding[i] = Math.Cos(angle);
        }
        else
        {
            angle = position / Math.Pow(10000, (double)(i - 1) / embeddingSize);
            positionalEncoding[i] = Math.Sin(angle);
        }

       
    }

    for (int i = 0; i < embeddingSize; i++)
    {
        wordEmbedding[i] += positionalEncoding[i];
    }

    return wordEmbedding;
}



double[] LinearLayer(double[] input, double[][] weights, double[] biases)
{
    int inputSize = input.Length;
    int outputSize = biases.Length;

    double[] output = new double[outputSize];

    for (int i = 0; i < outputSize; i++)
    {
        double sum = biases[i];
        for (int j = 0; j < inputSize; j++)
        {
            sum += input[j] * weights[i][j];
        }
        output[i] = sum;
    }

    return output;
}



public class Head
{
    public double[] Q { get; set; }
    public double[] K { get; set; }
    public double[] V { get; set; }
}