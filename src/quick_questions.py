"""
quick_questions.py — YKS Koçluk Soru Havuzu
=============================================
Sidebar hızlı sorularında ve hoş-geldin kartında kullanılan
üst-düzey koçluk senaryolarını içerir.

Dışa aktarılanlar:
  ALL_QUESTIONS  — 50 sorudan oluşan tam havuz (kategori sözlüğü ve düz liste)
  FLAT_LIST      — Düz liste; random.sample() ile kullanıma hazır
  SAMPLE_QUESTIONS — Hoş-geldin kartı için seçilmiş 6 temsili soru
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────
# 50 ÜSTÜN-DÜZEY YKS KOÇLUK SENARYOSU
# ─────────────────────────────────────────────────────────────

ALL_QUESTIONS: dict[str, list[str]] = {

    # ── MOTİVASYON & PSİKOLOJİ ───────────────────────────────
    "Motivasyon & Psikoloji": [
        "Öğrencim 3 aydır çalışmasına rağmen net artmıyor ve 'Boşuna çabalıyorum' diyor. "
        "Tükenmişlik sendromunu nasıl kırarım?",

        "Mezuna kalmış öğrencim Aralık ayında 'Yine olmayacak' krizine girdi. "
        "Bu noktada koçluk seansını nasıl tasarlamalıyım?",

        "Öğrencim sınav günü panik atak geçirdi ve TYT'yi yarıda bıraktı. "
        "Hem psikolojik toparlanma hem de kalan süre planlaması için ne önerirsiniz?",

        "Öğrencim sabah çalışmaya başlayamıyor, motivasyonu öğleden sonra geliyor. "
        "Biyolojik ritme göre program nasıl yapılandırmalıyım?",

        "Öğrencim sosyal medyayı bırakamıyor ve her gün 3-4 saat telefonda geçiriyor. "
        "Bu alışkanlığı koçluk perspektifinden nasıl kırabiliriz?",

        "Birden fazla öğrencim aynı hafta kötü deneme sonucu aldı ve moral bozuk. "
        "Grup motivasyonu için nasıl bir seans tasarlayayım?",

        "Öğrencim hedef belirsizliğinden muzdarip; 'Hangi bölümü istediğimi bilmiyorum' "
        "diyor. Meslek rehberliğini koçluk seansına nasıl entegre ederim?",
    ],

    # ── TYT STRATEJİSİ & KAYNAK ───────────────────────────────
    "TYT Strateji & Kaynak": [
        "TYT'de Türkçe'de dil bilgisi tam ama paragrafta süre kaybediyor. "
        "Hız + doğruluk dengesini kurmak için hangi taktiği önerirsiniz?",

        "TYT Matematik'te temel işlem hataları var; soru tipi anlaşılıyor ama "
        "hesap yanlış çıkıyor. Bunun için odaklı bir çalışma planı nedir?",

        "Öğrencim TYT'de 60 net bandına takıldı, 4 aydır geçemiyor. "
        "Hangi alt başlıkları öncelikli gözden geçirmeliyiz?",

        "TYT Fen'de Kimya ve Fizik çok düşük ama Biyoloji tam. "
        "Sınırlı sürede hangi konuları es geçip hangilerine odaklanmalı?",

        "TYT Sosyal'de Tarih çok zayıf. Kronoloji karmaşası var. "
        "Kısa sürede net artıracak Tarih çalışma yöntemi nedir?",

        "Öğrencim TYT'ye 6 hafta kaldığında başladı. 'Sadece TYT geçsem yeter' "
        "diyor. Gerçekçi bir 6 haftalık acil plan nasıl kurgulanır?",

        "TYT denemelerinde net tutarlı değil; bazen 80 bazen 50 çıkıyor. "
        "Bu dengesizliğin kök nedeni nasıl tespit edilir?",
    ],

    # ── AYT STRATEJİSİ & KAYNAK ───────────────────────────────
    "AYT Strateji & Kaynak": [
        "AYT Fen netleri 25'te takılı kaldı; Fizik yapamıyor ama Biyoloji güçlü. "
        "Ağırlıklı puana göre bir yönlendirme planı nasıl çizmeliyim?",

        "AYT Matematik'te Türev ve İntegral konularında sıfır net var. "
        "Bu konular için başlangıç-orta-ileri sıralı kaynak önerisi nedir?",

        "Öğrencim AYT Edebiyat'ta şiir sorularından sıfır alıyor. "
        "Şiir anlama becerisini geliştirmek için pratik bir yöntem nedir?",

        "AYT Geometri'de Çember ve Analitik konuları çok zayıf. "
        "Bu iki konuyu 3 haftada makul bir seviyeye taşıyabilir miyiz?",

        "Öğrencim AYT Sözel ağırlıklı bir bölüm hedefliyor ama Felsefe "
        "ve Din konularına hiç çalışmamış. Öncelik sırası nasıl olmalı?",

        "AYT'de iki farklı alan (Sayısal + Sözel) çalışan bir öğrencim var. "
        "Zamanı nasıl bölmeli, hangi alanı daha ağır tutmalı?",

        "Öğrencim AYT Fizik'te formülleri biliyor ama problem kuramıyor. "
        "Soru kurma becerisini geliştirmek için nasıl bir antrenman yapmalıyız?",
    ],

    # ── DENEME ANALİZİ & HATA YÖNETİMİ ──────────────────────
    "Deneme Analizi & Hata Yönetimi": [
        "Öğrencim deneme analizini çok yüzeysel yapıyor; sadece kaç net yaptığına "
        "bakıp geçiyor. Derin hata analizi için ne önerirsiniz?",

        "Hata defteri tutturmaya çalışıyorum ama öğrenci birkaç günde bırakıyor. "
        "Bu alışkanlığı kalıcı hâle getirmenin yolu nedir?",

        "Öğrencim denemede süre yetersizliği yaşıyor; soruları anlıyor ama "
        "yetiştiremiyoruz. Zaman yönetimi taktikleri nelerdir?",

        "Öğrencim aynı hataları tekrarlıyor. Hata defteri var ama okumayı "
        "ihmal ediyor. Bu döngüyü kırmak için ne yapmalıyım?",

        "Öğrencim TYT denemesinde Matematik'te 30 net yapıyor ama konu testlerinde "
        "28-30 alıyor. Deneme psikolojisi üzerine nasıl çalışmalıyız?",

        "Öğrencim boş bırakma korkusu yüzünden hepsini işaretliyor ve negatif "
        "yiyor. Risk analizi ve puan stratejisi nasıl öğretilir?",
    ],

    # ── PROGRAM & ZAMAN YÖNETİMİ ─────────────────────────────
    "Program & Zaman Yönetimi": [
        "Öğrencimin TYT'ye 5 ayı var ve her ders zayıf; sıfırdan başlıyor. "
        "Makul net hedefleriyle haftalık program nasıl tasarlanır?",

        "Öğrencim hem okul sınavlarına hem TYT'ye çalışıyor ve "
        "önceliklendirme yapamıyor. Bu çift yük nasıl yönetilir?",

        "Dersanede yoğun tempo var, öğrenci evde artık çalışmak istemiyor. "
        "Dershane + ev çalışmasını nasıl dengeleyeyim?",

        "Öğrencim hafta sonları tamamen çalışmıyor. Hafta sonu programını "
        "motivasyonu kırmadan nasıl kurgulayayım?",

        "Öğrencim konuları anlıyor ama soru çözmekten kaçıyor. Konu anlatımı "
        "ile soru çözüm dengesini nasıl kurmalıyım?",
    ],

    # ── VELİ & AİLE İLETİŞİMİ ────────────────────────────────
    "Veli & Aile İletişimi": [
        "Öğrencinin ailesi zorla Tıp istiyor ama öğrenci Mühendisliğe yatkın "
        "ve baskıdan bunalmış. Veli görüşmesini nasıl yönetmeliyim?",

        "Veli her hafta beni arayıp 'Neden net artmıyor?' diye baskı yapıyor. "
        "Veli beklentilerini nasıl gerçekçi zemine oturtabilirim?",

        "Öğrencinin annesi ders saatlerine müdahale ediyor ve süreci zorlaştırıyor. "
        "Sınır belirleme konuşmasını nasıl yapmalıyım?",

        "Velisi çok düşük bir bütçeyle çalışıyor; kaynak alamıyor. "
        "Ücretsiz/ekonomik alternatiflerle nasıl bir plan kurarım?",

        "Ayrılmış bir aile var ve her ebeveyn farklı bölüm hedefi koyuyor. "
        "Bu çelişen baskıyı öğrenciden nasıl uzaklaştırabilirim?",
    ],

    # ── KRİZ SENARYOLARI ──────────────────────────────────────
    "Kriz Senaryoları": [
        "Öğrencim sınava 3 gün kala 'Hiçbir şey bilmiyorum, girmeyeyim' dedi. "
        "Bu krizi son dakikada nasıl yönetirim?",

        "Öğrencim TYT'den 10 gün önce ağır bir grip geçirdi. "
        "Hem iyileşme sürecinde hem de kalan günlerde nasıl bir plan çizelim?",

        "Öğrencim iyi bir TYT puanı yaparken AYT'de çöktü ve moral bitti. "
        "İkinci sınav motivasyonu için en etkili yaklaşım nedir?",

        "Öğrencim sonuçları açıklandığında hayal kırıklığı yaşadı ve "
        "'Bir daha girmeyeceğim' dedi. Yeniden motivasyonu nasıl inşa ederim?",

        "Öğrencim sınav öncesi gecesi uyuyamıyor. Uyku hijyeni ve "
        "sınav günü hazırlığı için pratik önerileriniz nelerdir?",
    ],

    # ── BRANŞ-SPESİFİK TAKTİKLER ──────────────────────────────
    "Branş-Spesifik Taktikler": [
        "Geometride Üçgenler bitmiş ama Dörtgenler çok zayıf. "
        "Dörtgenler konusunda maksimum verim için kaynak ve sıra önerisi nedir?",

        "TYT Türkçe'de cümle tamamlama ve paragraf soruları düşük. "
        "Bu iki alt başlıkta hızlı net artırma taktiği nedir?",

        "AYT Kimya'da Mol kavramı ve Denge soruları çözülemiyor. "
        "Bu konular için adım adım öğretim planı nedir?",

        "Öğrencim TYT Matematik'te problemlere çok zaman harcıyor. "
        "Problemi 2 dakikada çözme refleksi nasıl kazandırılır?",

        "AYT Tarih'te Osmanlı Dönemi soruları çok zayıf. "
        "Kronolojik ezber yerine ilişkilendirme yöntemi nasıl uygulanır?",

        "TYT Fen'de birçok konu var ve hepsine zaman yok. "
        "Puana katkısı en yüksek konulara odaklanmak için öncelik sırası nedir?",
    ],
}

# ─────────────────────────────────────────────────────────────
# YARDIMCI LİSTELER
# ─────────────────────────────────────────────────────────────

# Tüm soruların düz listesi (random.sample için kullanılır)
FLAT_LIST: list[str] = [q for questions in ALL_QUESTIONS.values() for q in questions]

# Hoş-geldin kartında gösterilecek temsili 6 soru
SAMPLE_QUESTIONS: list[str] = [
    "TYT'de 60 net bandına takıldı, 4 aydır geçemiyor. Ne yapmalıyım?",
    "AYT Fen netleri 25'te takılı; Fizik yapamıyor ama Biyoloji güçlü.",
    "Öğrencim burnout yaşıyor, tükenmişlik sendromunu nasıl kırarım?",
    "Öğrencim deneme analizini yüzeysel yapıyor, derinleştirmek için ne önerirsiniz?",
    "Aile zorla Tıp istiyor, öğrenci Mühendislik istiyor — veli görüşmesini nasıl yöneteyim?",
    "Sınava 3 gün kala 'Hiçbir şey bilmiyorum, girmeyeyim' dedi. Bu krizi nasıl yönetirim?",
]
