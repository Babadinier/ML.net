using System;
using System.IO;
using Microsoft.ML;
using sar.poc.machineLearning.models;

namespace sar.poc.machineLearning
{
    class Program
    {
        private static string _appPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string TrainingDataLocation => Path.Combine(_appPath, "..", "..", "..", "datas", "train.csv");

        private static MLContext mlContext;
        private static PredictionEngine<Mail, MailPrediction> predEngine;
        private static ITransformer trainedModel;
        static IDataView trainingDataView;

        static void Main(string[] args)
        {
            mlContext = new MLContext(seed: 0);

            trainingDataView = mlContext.Data.LoadFromTextFile<Mail>(TrainingDataLocation, hasHeader: true, separatorChar: ';');

            var pipeline = ProcessData();

            var trainingPipeline = BuildAndTrainModel(trainingDataView, pipeline);

            Mail mail = new Mail { Subject = "Appels en absence", Body = "Vous avez manqué un appel de +33 1 47 40 84 05" };          
            
            Prediction(mail);
        }

        private static IEstimator<ITransformer> ProcessData()
        {
            Console.WriteLine($"=============== Processing Data ===============");
            
            // STEP 2: Common data process configuration with pipeline data transformations
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Result", outputColumnName: "Label")
                            .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Subject", outputColumnName: "SubjectFeaturized"))
                            .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Body", outputColumnName: "BodyFeaturized"))
                            .Append(mlContext.Transforms.Concatenate("Features", "SubjectFeaturized", "BodyFeaturized"))
                            .AppendCacheCheckpoint(mlContext);

            Console.WriteLine($"=============== Finished Processing Data ===============");
            
            return pipeline;
        }

        private static IEstimator<ITransformer> BuildAndTrainModel(IDataView trainingDataView, IEstimator<ITransformer> pipeline)
        {
            // STEP 3: Create the training algorithm/trainer
            // Use the multi-class SDCA algorithm to predict the label using features.
            //Set the trainer/algorithm and map label to value (original readable state)
            var trainingPipeline = pipeline.Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // STEP 4: Train the model fitting to the DataSet
            Console.WriteLine($"=============== Training the model  ===============");

            trainedModel = trainingPipeline.Fit(trainingDataView);
            Console.WriteLine($"=============== Finished Training the model Ending time: {DateTime.Now.ToString()} ===============");

            return trainingPipeline;
        }

        private static void Prediction(Mail mail)
        {
            Console.WriteLine($"=============== Single Prediction just-trained-model ===============");

            // Create prediction engine related to the loaded trained model
            predEngine = mlContext.Model.CreatePredictionEngine<Mail, MailPrediction>(trainedModel);

            var prediction = predEngine.Predict(mail);

            Console.WriteLine($"=============== Single Prediction just-trained-model - Result: {prediction.Result} ===============");
        }
    }
}
