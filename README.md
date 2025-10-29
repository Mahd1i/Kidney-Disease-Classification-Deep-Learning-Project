# Kidney-Disease-Classification-Deep-Learning-Project

## تصنيف أمراض الكلى باستخدام التعلم العميق (VGG16 Transfer Learning)

هذا المشروع يهدف إلى بناء نموذج لتصنيف صور الكلى إلى فئتين رئيسيتين: **Normal (سليمة)** و **Tumor/Cancer (ورم/سرطان)** باستخدام نموذج VGG16 المُدرَّب مسبقاً (Transfer Learning). تم بناء المشروع وتتبع تجاربه بالكامل باستخدام منهجية MLOps (**MLflow & DVC**).

---

## 🏆 حالة المشروع النهائية

* **النموذج الأقوى:** VGG16Model v4
* **دقة التقييم:** 100% على مجموعة التقييم
* **منصة MLOps:** Dagshub (للتتبع، البيانات، والنموذج)
* **ملاحظات:** لم يتم إعادة تدريب النموذج أثناء التقييم الأخير؛ تم فقط تقييم النموذج الموجود مسبقًا وحفظ النتائج.

---

## Workflows (مسار العمل المُكتمل)

1. Update config.yaml (تحديث ملفات الإعداد)
2. Update secrets.yaml [Optional] (تحديث المتغيرات الحساسة)
3. Update params.yaml (تحديث المعاملات التشعبية)
4. Update the entity (تحديث كيانات البيانات)
5. Update the configuration manager in src config (تحديث مدير الإعداد)
6. Update the components (تحديث المكونات: جلب البيانات، التدريب، التقييم، إلخ)
7. Update the pipeline (بناء خط الأنابيب)
8. Update the main.py (نقطة دخول التشغيل)
9. Update the dvc.yaml (تحديث خطوات DVC)
10. app.py (نشر الويب الأولي)
11. Update the prediction component and test with dvc_predict.py (مرحلة التنبؤ والاختبار النهائي)

---

## كيفية التشغيل (How to Run)

### 1️⃣ استنساخ المستودع (Clone the repository)

```bash
git clone https://github.com/Mahd1i/Kidney-Disease-Classification-Deep-Learning-Project
cd Kidney-Disease-Classification-Deep-Learning-Project
```

### 2️⃣ إنشاء بيئة Conda

```bash
conda create -n kidney_env python=3.10 -y
conda activate kidney_env
```

### 3️⃣ تثبيت المتطلبات

```bash
pip install -r requirements.txt
pip install -e .
```

### 4️⃣ تشغيل التطبيق

```bash
python app.py
```

ثم افتح المتصفح على `localhost` مع المنفذ المحدد في التطبيق.

---

## 🧰 MLflow & DVC

### MLflow

* لتتبع جميع التجارب والنماذج
* تسجيل النتائج والمعاملات (params & metrics)
* نشر النموذج على Dagshub

### DVC

* متتبع تجارب خفيف وفعال
* إنشاء خطوط أنابيب reproducible
* دعم orchestration للبيانات والنماذج

---

### 5️⃣ إعداد MLflow مع Dagshub

قبل تشغيل أي سكربت، قم بتعيين المتغيرات البيئية في الـ CMD (أو PowerShell):

```bash
set MLFLOW_TRACKING_URI="https://dagshub.com/Mahd1i/Kidney-Disease-Classification-Deep-Learning-Project.mlflow"
set MLFLOW_TRACKING_USERNAME=Mahd1i
set MLFLOW_TRACKING_PASSWORD=YOUR_PERSONAL_ACCESS_TOKEN_HERE
```

> 💡 ملاحظة: استبدل `YOUR_PERSONAL_ACCESS_TOKEN_HERE` بالـ Personal Access Token الخاص بك على Dagshub.

---

### 6️⃣ أوامر DVC المهمة

```bash
dvc init             # تهيئة DVC
dvc repro            # إعادة تشغيل خط الأنابيب بالكامل
dvc dag              # عرض خط الأنابيب
```

---

### 7️⃣ مصادر تعليمية

* [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
* [MLflow Tutorial (YouTube)](https://youtu.be/qdcHHrsXA48?si=bD5vDS60akNphkem)
