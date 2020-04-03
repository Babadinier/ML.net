using Microsoft.ML.Data;

namespace sar.poc.machineLearning.models
{
    public class MailPrediction
    {
        [ColumnName("PredictedLabel")]
        public string Result;
    }
}