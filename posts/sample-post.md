# Kendi yapay zeka modelimi nasıl geliştirdim?

## Giriş

Yapay zeka iş geliştirme süreçlerinde ve daha birçok işlem için kullanılıyor. Ve şu anlaşılıyor ki artık görmezden gelinecek bir teknoloji olmaktan çıktı. Öyle ki yapay zekayı hiç olmazsa kendi süreçlerinde kullanmayan veya özellikle biz geliştiriciler için yapay zekayı göz ardı ettiğimizde çağın gerisinde kalıyoruz desem -bu teknolojinin geldiği nokta düşünüldüğünde- abartmış olmam. Elbette yapay zeka kendi içinde derya deniz, çok fazla dalı ve çok fazla türü var. Doğal dil işlemeden tutun da görsel işlemeye kadar birçok alanı karşımıza çıkıyor. Hatta öğrenme süreçlerinde bile dallanmaya rastlamak mümkün ( Makine Öğrenmesi (ML),Derin Öğrenme (DL) vs.) 

Ben de bu gece merakıma yenik düşüp “ Kendi yapay zeka modelimi geliştirebilir miyim?” sorusuna yanıt aradım. Amatörce bile olsa kendi minik modelimi eğitmeyi başardım. Sonuçlar her ne kadar çok da tutarlı olmasa da anladım ki kendimi bu konuda derinlemesine geliştirdiğim taktirde bir geliştirici olarak kendi uygulamalarımda herhangi bir 3. parti yapay zeka modeli kullanmak yerine ( ChatGPT, Cloude, Gemini vs ) basit çıktılar için kendi modelimi eğitip uygulamalarıma entegre edebileceğimi gördüm. 

Eğer sizler de kendi minik modellerinizi eğitmek istiyorsanız endişelenmeyin, başlangıç seviyede kütüphaneler yardımıyla çok kolay bir şekilde kendi amacına yönelik modelinizi eğitebilirsiniz. Bu yazıyı yazmamdaki amaç bu başlangıcı nasıl yapabileceğinize dair size bir yol göstermek. Aşağıda adım adım **“Ürün yorumuna göre puan tahminlemesi yapan (5 üzerinden) bir yapay zeka modelini”** nasıl geliştirdiğimi anlatıcağım. Siz farklı verilerle farklı amaçlara yönelik minik yapay zeka modelleri geliştirebilirsiniz.

## Adım 1: Veri setini elde etme ve Veriyi eğitime uygun hale getirme

Yapay zeka modelleri için veri çok kritik bir öneme sahip. Öyle ki eğitimde kullanacağımız veriler ne kadar tutarlı ve çeşitli olursa modelimiz o kadar doğruya yakın çıktılar üretebilirler. Bizim de en çok üzerine düşmemiz gereken adım bu; amaca uygun veriyi elde etme ve o veriyi yine amaca uygun eğitimde kullanılabilir hale getirme. Kaggle ve HuggingFace gibi platformlar herkese açık bir şekilde amacına göre veri setleri yayınlıyor ancak ben **ürün yorumunun puanını tahminleyen** bir yapay zeka modeli geliştirmek istediğimden ve üstelik türkçe yorumları tahminlemesini gerektiğinden dolayı bir e ticaret sitesinin yorumlarını **web scrapping**  yoluyla çekip json dosyası içerisinde bu verileri toparladım. Bu yönülye web scrapping de veri seti elde etmede en etkili yöntem olarak karşımıza çıkıyor. 

Bir şekilde istediğimiz verileri topladık diyelim. Şimdi ise düşünmemiz gereken başka bir şey var; **modelime hangi girdileri vereceğim ve hangi çıktıları üretmesini istiyorum?** Örnek olması açısından kendi modelim için **girdi: ürün yorumları ve çıktı: puan ( 5 üzerindein )** olmalı. Yani ben bir yorum yazacağım, modelim o yorumu yazan kişinin tahmini ne kadar puan verdiğini benim için tahminleyecek. İşte burdan yola çıkacak olursak bizim eğitim için kullanacağımız veriler aşığdaki yapıya benzemelidir;

```json
[{
    "star": 1,
    "content": "Sipariş ettiğim ürün baştaki gönderdikleri ürün farklı renk tamamen alakasız. Erkek arkadaşıma doğum günü hediyesi almıştım mahcup oldum teşekkür ederim"
  },
  {

    "star": 3,
    "content": "Ürün kumaşı iyi ancak 115 kg olmama rağmen göbek kısmı dar, bel çevresi de sıktığı için iade ettim boyu uzun. Slim fit ürünlerde kilolu olanlar rahat etmesi için 3x veya xxxl alınabilir"
  },
  {
    "star": 1,
    "content": "Kargom teslim edildi ama sayfaya girdiğimde teslim edilmemiş olarak gözüküyor ve iade ettim kargoda iade etmiş oldum ve hepsi jet kapıdan alıyor mu (ürün orhinal değildi boş yere uğraştım)geri dönüşünü beklerim.."
  },
  ...
```

Bir diğer üzerinde durmamız gerekn konu **verinin ne kadar tutarlı** olduğunu anlamamızdır. Boş veri veya bozuk veri var mı? İçerisinde istediğimiz aralıkların dışına çıkan veriler bulunuyor mu? Bunun gibi soruları da cevap vermek lazım. Benim kendi eğitim verim için örneğin yıldız dağılımı nasıl, biri diğerinden çok mu fazla veya çok mu az? Yorumların uzunluğu ne kadar mesela? Bu gibi soruları ne kadar genişletirsek verilerimiz de eğitim için çok daha verimli hale gelir. Örnek olması açısından bunu kontrol eden bir python fonksiyonu yazdım. Bu fonksiyon ayrıca grafik çıktıları da vererek veriyi görsel olarak daha iyi anlamamı sağlıyor.

```python
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt

def veri_kalite_raporu(json_path):
    # JSON verisini oku
    df = pd.read_json(json_path)

    print("Sınıf Dağılımı")
    print("="*40)
    print(df['star'].value_counts(), "\n")

    # Eksik veri kontrolü
    print("Eksik Veri Kontrolü")
    print("="*40)
    print(df.isnull().sum(), "\n")

    # Boş içerik kontrolü
    bos_yorumlar = df['content'].str.strip().eq('').sum()
    print(f"Boş yorum sayısı: {bos_yorumlar}\n")

    # Yorum uzunlukları
    df['length'] = df['content'].str.len()
    print("Yorum uzunluğu (min/ort/medyan/max):")
    print(df['length'].describe(), "\n")

    # Özel karakter kontrolü
    ornek_ozel = df[df['content'].str.contains(r'[^\w\s]', na=False)].head(3)['content']
    print("Özel karakter içeren örnek yorumlar:")
    print(ornek_ozel, "\n")

    # Duplicates
    dupe_count = df.duplicated(subset=['content']).sum()
    print(f"Tekrarlayan yorum sayısı: {dupe_count}\n")

    # Sınıf dağılım grafiği
    sns.countplot(x='star', data=df, palette='Set2')
    plt.title("Yıldız Dağılımı")
    plt.show()

    # Yorum uzunluğu dağılımı
    df['length'].hist(bins=50)
    plt.title("Yorum Uzunluğu Dağılımı")
    plt.xlabel("Karakter sayısı")
    plt.ylabel("Adet")
    plt.show()

    return df

```

```
Sınıf Dağılımı
========================================
star
3    1259
5     688
1     607
2     162
4      97
Name: count, dtype: int64 

Eksik Veri Kontrolü
========================================
code        0
star        0
content    97
dtype: int64 

Boş yorum sayısı: 0

Yorum uzunluğu (min/ort/medyan/max)
count    2716.000000
mean      196.757364
std       203.611658
min         3.000000
25%        77.000000
50%       143.000000
75%       245.000000
max      1560.000000
Name: length, dtype: float64 

Özel karakter içeren örnek yorumlar:
0    Ürün güzel kalın.. 17 aylık oğluma 2 4 yaş ald...
1    Boyları renkler arasında farklılık gösteriyor ...
2    Çok güzel zaten alışveriş yaptığım bi firma be...
Name: content, dtype: object 

Tekrarlayan yorum sayısı: 1362
```

![image.png](posts/image.png)

![image.png](posts/image%201.png)

## Adım 2: Algoritma seçimi, Modeli eğitme ve skorları görüntüleme

Veriler için gerekli düzenlemeleri yaptığımızı ve nihayi eğitim verisini elde ettiğimizi varsayıyorum. Şimdi sıra bu veriler ile kendi modelimizi eğitmekte. Makine öğrenmesinde bir veriyi eğitmek için birçok algoritma bulunuyor. Bizim kullanacağmız algoritma azınlık verilerimizinde olmasından ötürü **Balanced Random Forest** algoritması olacak. Bu sırada **SMOTE** ile azınlıkta olan verilerimizi yeni sentetik veriler üreterek çoğunluğa biraz daha yaklaştırıp tutarlı sonuçlar almayı amaçlayacağız.

```python
# Gerekli kütüphaneler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
import joblib

# JSON dosyasını oku

# Metni sayısallaştır (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['content'])
y = df['star']

# Eğitim ve test seti ayır
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# SMOTE ile azınlık sınıflarını oversample et
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Balanced Random Forest sınıflandırıcı
brf = BalancedRandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# Modeli eğit
brf.fit(X_train_res, y_train_res)

# Test setinde tahmin ve rapor
y_pred = brf.predict(X_test)
report = classification_report(y_test, y_pred, digits=4)

# Sınıf bazlı precision, recall, f1-score grafiği
report_dict = classification_report(y_test, y_pred, output_dict=True)
classes = sorted([c for c in report_dict.keys() if c.isdigit()])

precision = [report_dict[c]['precision'] for c in classes]
recall    = [report_dict[c]['recall'] for c in classes]
f1_score  = [report_dict[c]['f1-score'] for c in classes]

x = np.arange(len(classes))
width = 0.25

fig, ax = plt.subplots(figsize=(10,6))
rects1 = ax.bar(x - width, precision, width, label='Precision', color='skyblue')
rects2 = ax.bar(x, recall, width, label='Recall', color='lightgreen')
rects3 = ax.bar(x + width, f1_score, width, label='F1-score', color='salmon')

ax.set_xlabel('Sınıflar', fontsize=12)
ax.set_ylabel('Skor', fontsize=12)
ax.set_title('Sınıf Bazlı Precision, Recall ve F1-Score', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.set_ylim(0, 1.05)
ax.legend()

for rects in [rects1, rects2, rects3]:
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

plt.show()

```

![image.png](posts/image%202.png)

Bu grafiği analiz edebilmemiz için parametrelerin ne anlama geldiğini bilmemiz gerekiyor;

**Precision:** Kesinlik. Yani örnek olarak modelin“bu yorum 5 yıldızdır” dediğinde, gerçekten 5 yıldız olma olasılığını gösterir.

**Recall:** Duyarlılık. Örnek olarak Gerçekten 5 yıldız olan yorumların ne kadarını model yakalayabildiğinin metriğini gösteriri

**F1-Score:**  Precision ve Recall parametrelerin dengeli ortalaması.

Yani yukarıdaki grafiği yorumlamak istersek örneğin yorumun 3 yıldız olduğunu diğer gruplara göre daha kesin olarak anladığını ve modelin bunu daha çok yakaladığını söyleyebiliriz. Öte yandan yorumun 4 yıldız olduğunu model daha çok yakalayabilmesine rağmen gerçekten 4 yıldız olduğunu 3 yıldız olduğunu kesin olarak bildiği kadar bilemediğini söyleyebiliriz.

## Adım 3: Modeli kaydetme ve örnek bir uygulamada kullanma

Modeli eğittikten sonra ilk işim onu kaydetmek oldu. Bunun için Python’da çok sık kullanılan **joblib** kütüphanesini kullandım. Böylece modeli her seferinde sıfırdan eğitmek zorunda kalmadan, hazır şekilde yükleyip kullanabiliyorum. Yalnızca modeli değil, metinleri sayısal verilere dönüştüren **vectorizer** nesnesini de sakladım. Çünkü yeni bir yorumu tahmin etmek istediğimde, önce bu vectorizer sayesinde metin sayısallaştırılıyor, ardından model puanı tahmin ediyor.

Sonraki adımda bu modeli nasıl daha pratik hale getirebilirim diye düşündüm ve Flask ile küçük bir web servisi geliştirdim. Bu servis çok basit çalışıyor: Kullanıcıdan gelen yorumu alıyor, vectorizer ile dönüştürüyor ve modeli kullanarak tahmini yıldız puanını geri döndürüyor. Bununla yetinmeyip küçük bir HTML arayüz de ekledim. Böylece kullanıcı yorumunu yazıyor ve sonucu ekranda yıldız ikonlarıyla görebiliyor.

```python
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Model ve vectorizer
model = joblib.load("yorum_modeli_brf.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    tahmin = None
    yorum = ""
    if request.method == "POST":
        yorum = request.form.get("yorum", "")
        if yorum:
            X = vectorizer.transform([yorum])
            tahmin = int(model.predict(X)[0])
    return render_template("index.html", tahmin=tahmin, yorum=yorum)

if __name__ == "__main__":
    app.run(port=5000, debug=True)

```

```html
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <title>Yorum Puanı Tahmini</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f5f5f5;
            display: flex;
            justify-content: center;
            padding-top: 50px;
        }
        .container {
            background-color: #fff;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 6px 15px rgba(0,0,0,0.1);
            width: 500px;
            text-align: center;
        }
        textarea {
            width: 100%;
            height: 100px;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #ccc;
            resize: none;
            font-size: 16px;
        }
        button {
            margin-top: 15px;
            padding: 12px 25px;
            border: none;
            border-radius: 8px;
            background-color: #007bff;
            color: #fff;
            font-size: 16px;
            cursor: pointer;
        }
        button:hover {
            background-color: #0056b3;
        }
        .stars {
            font-size: 32px;
            color: #ffbf00;
            margin-top: 15px;
        }
        .star.empty {
            color: #ddd;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Yorum Puanı Tahmini</h1>
        <form method="post">
            <textarea name="yorum" placeholder="Yorumunuzu buraya yazın...">{{ yorum }}</textarea>
            <br>
            <button type="submit">Tahmin Et</button>
        </form>

        {% if tahmin is not none %}
            <div class="stars">
                {% for i in range(tahmin) %}
                    ★
                {% endfor %}
                {% for i in range(5 - tahmin) %}
                    <span class="star empty">☆</span>
                {% endfor %}
            </div>
        {% endif %}
    </div>
</body>
</html>

```

## Sonuç

Bu çalışma boyunca bir yapay zeka modelinin geliştirilme sürecini baştan sona deneyimleme fırsatı buldum. Verilerin toplanması, temizlenmesi ve uygun forma getirilmesi aşamalarının, modelin başarısı üzerinde ne kadar büyük bir etkisi olduğunu gördüm. Basit gibi görünen ama aslında kritik olan veri ön işleme adımları, elde edilen sonuçların kalitesini doğrudan belirledi.

Eğitim aşamasında Balanced Random Forest gibi dengesiz veri setleriyle başa çıkabilen bir algoritmayı tercih ederek sınıflar arasında daha dengeli bir performans elde ettim. Yine de bazı sınıflarda precision ve recall değerlerinin düşük kaldığını gözlemledim. Bu durum, daha fazla ve daha temiz veriye ihtiyaç duyulduğunu açıkça gösteriyor.

Modeli kaydedip Flask aracılığıyla bir API ve basit bir arayüz geliştirmek ise bu sürecin en keyifli kısmı oldu. Böylece, yalnızca teorik bir modelden öte, kullanıcıların etkileşime girebileceği küçük ama işlevsel bir uygulama ortaya çıkmış oldu.

Sonuç olarak, bu proje bana yapay zekânın sadece “model eğitmek” olmadığını, aynı zamanda veriyle doğru çalışmayı, modeli uygun şekilde değerlendirmeyi ve onu pratik bir uygulamaya dönüştürmeyi öğretti. Gelecekte daha büyük ve çeşitli veri setleriyle, BERT gibi transformer tabanlı modern modelleri de deneyerek bu çalışmayı bir adım öteye taşımayı hedefliyorum.