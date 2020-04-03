using Microsoft.ML.Data;

namespace sar.poc.machineLearning.models
{
    public class Mail 
    {
        [LoadColumn(0)]
        public string Subject;

        [LoadColumn(1)]
        public string Body;

        [LoadColumn(2)]
        public string Result;
    }
}