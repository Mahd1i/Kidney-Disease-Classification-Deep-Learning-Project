import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import read_yaml, create_directories,save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    
    def _valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )


    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    def log_into_mlflow(self):
        import os

        # 🟢 إعداد بيانات الدخول إلى Dagshub
        import os

        os.environ["MLFLOW_TRACKING_USERNAME"] = "Mahd1i"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = "23a0d017c3731bc195e9683c8bc813849992b5f7"

        # 🔹 إذا كنت تستخدم Dagshub (سيرفر أونلاين)
        if self.config.mlflow_uri.startswith("http"):
            mlflow.set_tracking_uri(self.config.mlflow_uri)
            print(f"🔗 Using remote MLflow tracking URI: {self.config.mlflow_uri}")
        else:
            # 🔹 استخدام تخزين محلي آمن (بدون file:)
            tracking_dir = os.path.join(os.getcwd(), "mlruns")
            os.makedirs(tracking_dir, exist_ok=True)
            mlflow.set_tracking_uri(tracking_dir)
            print(f"💾 Using local MLflow tracking directory: {tracking_dir}")

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )

            # 🔹 رفع النموذج إلى MLflow
            mlflow.keras.log_model(self.model, artifact_path="model")

            print("\n✅ Model logged successfully to MLflow!")

