import os
import shutil
import random
from pathlib import Path

# افتراض أنك تحتاج إلى تحديد مسارات الإدخال والإخراج هنا، 
# أو قراءة الإعدادات من ConfigurationManager كما يفعل المدرب.

# بما أننا لا نملك ملف config.yaml، سنفترض المسارات ونستخدم مكتبات بايثون القياسية
# (في مشروعك الفعلي، يجب أن تستخدم ConfigurationManager)

STAGE_NAME = "Data Split Stage"
ROOT_DIR = Path("artifacts/data_ingestion/kidney-ct-scan-image")

# نسبة التقسيم (مثال: 80% تدريب، 20% تحقق)
TRAIN_RATIO = 0.8
VALIDATION_RATIO = 0.2

def split_data(root_dir: Path, train_ratio: float, validation_ratio: float):
    """يقوم بتقسيم الصور الموجودة مباشرة داخل مجلدات الفئات إلى مجلدات train و validation"""
    
    # 1. تحديد مسارات الإدخال والإخراج
    output_dir = root_dir
    train_dir = output_dir / "train"
    validation_dir = output_dir / "validation"
    
    # قائمة بأسماء الفئات (Normal, Tumor)
    class_names = [d.name for d in root_dir.iterdir() if d.is_dir() and d.name not in ['train', 'validation']]

    if not class_names:
        print(f"ERROR: No class directories found in {root_dir}")
        return

    # 2. إنشاء مجلدات الإخراج
    for d in [train_dir, validation_dir]:
        d.mkdir(parents=True, exist_ok=True)
        for class_name in class_names:
            (d / class_name).mkdir(parents=True, exist_ok=True)

    # 3. معالجة وتقسيم كل فئة
    print(f"Starting data split for classes: {class_names}")
    for class_name in class_names:
        class_path = root_dir / class_name
        files = [f for f in class_path.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        # خلط ترتيب الملفات
        random.shuffle(files)
        
        num_files = len(files)
        num_train = int(train_ratio * num_files)
        
        train_files = files[:num_train]
        validation_files = files[num_train:]
        
        print(f"  - Class '{class_name}': Total={num_files}, Train={len(train_files)}, Validation={len(validation_files)}")

        # 4. نقل الملفات
        # نقل إلى مجلد التدريب
        for f in train_files:
            shutil.move(str(f), str(train_dir / class_name / f.name))
        
        # نقل إلى مجلد التحقق
        for f in validation_files:
            shutil.move(str(f), str(validation_dir / class_name / f.name))
            
        # 5. حذف مجلد الفئة الأصلي الفارغ بعد نقل جميع محتوياته
        class_path.rmdir() 
    
    print("Data split complete. Data is ready for training.")


def main():
    # في مشروعك الفعلي، يمكنك استخدام ConfigurationManager هنا إذا كان لديك إعدادات للتقسيم.
    # config = ConfigurationManager()
    # split_config = config.get_data_split_config()
    # split_data(Path(split_config.root_dir), split_config.train_ratio, split_config.validation_ratio)
    
    # التنفيذ باستخدام القيم الثابتة أعلاه
    split_data(ROOT_DIR, TRAIN_RATIO, VALIDATION_RATIO)


if __name__ == '__main__':
    try:
        print(f"*******************")
        print(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        main()
        print(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        print(f"Error in {STAGE_NAME}: {e}")
        raise e