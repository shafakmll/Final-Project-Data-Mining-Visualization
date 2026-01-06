
# ---------------------------
# Install & load packages if missing
# ---------------------------
pkgs <- c("shiny","shinydashboard","DT","shinycssloaders","randomForest","dplyr","ggplot2","patchwork","tidyr","naniar","vcd","reshape2","pROC")
for (p in pkgs) if(!requireNamespace(p, quietly = TRUE)) install.packages(p, dependencies = TRUE)

library(shiny)
library(shinydashboard)
library(DT)
library(shinycssloaders)
library(randomForest)
library(dplyr)
library(ggplot2)
library(patchwork)
library(tidyr)
library(naniar)
library(vcd)
library(reshape2)
library(pROC)

# ---------------------------
# User dataset path (adjust if needed)
# ---------------------------
data_path <- "mushrooms.csv"
data <- read.csv(data_path, stringsAsFactors = FALSE)
head(data)
# Normalize class column
if("class" %in% names(data) && !("Class" %in% names(data))){
  names(data)[names(data) == "class"] <- "Class"
}
# Ensure factor with labels edible/poisonous
if(!is.factor(data$Class)){
  data$Class <- factor(data$Class, levels = c("e","p"), labels = c("edible","poisonous"))
}

# ---------------------------
# Category mapping (provided)
# ---------------------------
cat_map <- list(
  gill.color = c("black"="k","brown"="n","buff"="b","chocolate"="h","gray"="g","green"="r","orange"="o","pink"="p","purple"="u","red"="e","white"="w","yellow"="y"),
  ring.type = c("evanescent"="e","flaring"="f","large"="l","none"="n","pendant"="p"),
  spore.print.color = c("black"="k","brown"="n","buff"="b","chocolate"="h","green"="r","orange"="o","purple"="u","white"="w","yellow"="y"),
  gill.size = c("broad"="b","narrow"="n"),
  bruises = c("bruises"="t","no bruises"="f"),
  stalk.surface.above.ring = c("fibrous"="f","scaly"="y","silky"="k","smooth"="s"),
  odor = c("almond"="a","anise"="l","creosote"="c","fishy"="y","foul"="f","musty"="m","none"="n","pungent"="p","spicy"="s"),
  stalk.root = c("bulbous"="b","club"="c","cup"="u","equal"="e","rhizomorphs"="z","rooted"="r","missing"="?"),
  stalk.surface.below.ring = c("fibrous"="f","scaly"="y","silky"="k","smooth"="s"),
  habitat = c("grasses"="g","leaves"="l","meadows"="m","paths"="p","urban"="u","waste"="w","woods"="d"),
  population = c("abundant"="a","clustered"="c","numerous"="n","scattered"="s","several"="v","solitary"="y"),
  gill.spacing = c("close"="c","crowded"="w","distant"="d")
)

top12_features <- names(cat_map)

# ---------------------------
# Helper to match feature keys to actual column names (handles -, ., _ variants)
# ---------------------------
find_colname <- function(key, df_names){
  if(key %in% df_names) return(key)
  alt1 <- gsub("\\.", "-", key); if(alt1 %in% df_names) return(alt1)
  alt2 <- gsub("-", ".", key); if(alt2 %in% df_names) return(alt2)
  alt3 <- gsub("\\.", "_", key); if(alt3 %in% df_names) return(alt3)
  alt4 <- gsub("-", "_", key); if(alt4 %in% df_names) return(alt4)
  lc <- tolower(key); matches <- df_names[tolower(df_names) == lc]; if(length(matches)) return(matches[1])
  return(NA_character_)
}
feature_col_map <- sapply(top12_features, find_colname, df_names = names(data), USE.NAMES = TRUE, simplify = TRUE)
available_features <- names(feature_col_map[!is.na(feature_col_map)])

# Convert character columns to factor for plotting
for(col in names(data)){
  if(!is.factor(data[[col]]) && is.character(data[[col]])) data[[col]] <- as.factor(data[[col]])
}

# ---------------------------
# Build Random Forest model (using available features)
# ---------------------------
predictors <- unname(unlist(feature_col_map[available_features]))
if(length(predictors) == 0) stop("Tidak menemukan fitur top12 pada dataset. Periksa nama kolom CSV.")
rf_formula <- as.formula(paste("Class ~", paste(predictors, collapse = "+")))
rf_model <- randomForest(rf_formula, data = data, ntree = 100, na.action = na.omit)

# ---------------------------
# Full feature rank table (from your data)
# ---------------------------
full_feature_rank <- data.frame(
  Feature = c("gill-color","ring-type","spore-print-color","gill-size","bruises",
              "stalk-surface-above-ring","odor","stalk-root","stalk-surface-below-ring",
              "habitat","population","gill-spacing","stalk-color-above-ring","stalk-color-below-ring",
              "cap-surface","cap-shape","ring-number","cap-color","stalk-shape","veil-color",
              "gill-attachment","veil-type"),
  MI = c(0.289027,0.220436,0.333199,0.159531,0.133347,0.197357,0.628043,0.093448,0.188463,0.108709,0.139987,0.069927,0.175952,0.167337,0.019817,0.033823,0.026653,0.024987,0.005210,0.016509,0.009818,0.000000),
  ChiSquare = c(5957.764469,1950.610146,379.132729,1636.606833,1194.277352,222.982400,75.910163,1186.029221,206.648180,751.309489,311.766736,826.795274,119.792216,109.789410,214.068544,17.508364,25.646335,11.511382,36.594105,5.126826,3.505447,NaN),
  Rank = c(2.0,3.0,5.0,6.0,7.5,7.5,8.0,9.0,9.0,9.5,9.5,10.0,10.0,11.0,14.5,16.5,16.5,18.0,18.5,19.5,20.5,22.0),
  stringsAsFactors = FALSE
)
feature_rank <- full_feature_rank %>% filter(Feature %in% top12_features)

# ---------------------------
# Model performance summary (user-supplied)
# ---------------------------
# ---------------------------
# Model performance summary (fixed: all use Holdout & KFold)
# ---------------------------
model_results <- data.frame(
  Model = c("Logistic Regression","Random Forest","XGBoost"),
  
  Holdout_Acc      = c(0.934003, 1.000000, 1.000000),
  KFold_Acc        = c(0.932420, 1.000000, 1.000000),
  
  Holdout_Precision = c(0.946113, 1.000000, 1.000000),
  KFold_Precision   = c(0.946113, 1.000000, 1.000000),
  
  Holdout_Recall    = c(0.915234, 1.000000, 1.000000),
  KFold_Recall      = c(0.915234, 1.000000, 1.000000),
  
  Holdout_F1        = c(0.930373, 1.000000, 1.000000),
  KFold_F1          = c(0.930373, 1.000000, 1.000000),
  
  Mean_AUC          = c(0.975675, 1.000000, 1.000000),
  
  stringsAsFactors = FALSE
)

# ---------------------------
# Variable explanation table (full 22 variables, Indonesian + code meanings)
# ---------------------------
vars_desc <- data.frame(
  Variable = c(
    "class","cap-shape","cap-surface","cap-color","bruises","odor","gill-attachment",
    "gill-spacing","gill-size","gill-color","stalk-shape","stalk-root",
    "stalk-surface-above-ring","stalk-surface-below-ring","stalk-color-above-ring",
    "stalk-color-below-ring","veil-type","veil-color","ring-number","ring-type",
    "spore-print-color","population","habitat"
  ),
  
  Meaning = c(
    "Jenis jamur (label target).",
    "Bentuk topi jamur.",
    "Permukaan topi jamur.",
    "Warna topi jamur.",
    "Apakah topi menghitam saat ditekan.",
    "Bau jamur.",
    "Cara insang menempel pada batang.",
    "Jarak antar insang.",
    "Ukuran insang.",
    "Warna insang.",
    "Bentuk batang.",
    "Bentuk akar batang.",
    "Tekstur batang di atas cincin.",
    "Tekstur batang di bawah cincin.",
    "Warna batang di atas cincin.",
    "Warna batang di bawah cincin.",
    "Jenis selubung (veil).",
    "Warna veil.",
    "Jumlah cincin pada batang.",
    "Tipe cincin pada batang.",
    "Warna spora (hasil cetakan spora).",
    "Kepadatan populasi jamur.",
    "Habitat tempat jamur ditemukan."
  ),
  
  Codes = c(
    "e, p",
    "b, c, x, f, k, s",
    "f, g, y, s",
    "n, b, c, g, r, p, u, e, w, y",
    "t, f",
    "a, l, c, y, f, m, n, p, s",
    "a, d, f, n",
    "c, w, d",
    "b, n",
    "k, n, b, h, g, r, o, p, u, e, w, y",
    "e, t",
    "b, c, u, e, z, r, ?",
    "f, y, k, s",
    "f, y, k, s",
    "n, b, c, g, o, p, e, w, y",
    "n, b, c, g, o, p, e, w, y",
    "p, u",
    "n, o, w, y",
    "n, o, t",
    "c, e, f, l, n, p, s, z",
    "k, n, b, h, g, r, o, p, u, e, w, y",
    "a, c, n, s, v, y",
    "g, l, m, p, u, w, d"
  ),
  
  Code_Meaning = c(
    "e = edible; p = poisonous",
    "b = bell; c = conical; x = convex; f = flat; k = knobbed; s = sunken",
    "f = fibrous; g = grooves; y = scaly; s = smooth",
    "n = brown; b = buff; c = cinnamon; g = gray; r = green; p = pink; u = purple; e = red; w = white; y = yellow",
    "t = bruises; f = no bruises",
    "a = almond; l = anise; c = creosote; y = fishy; f = foul; m = musty; n = none; p = pungent; s = spicy",
    "a = attached; d = descending; f = free; n = notched",
    "c = close; w = crowded; d = distant",
    "b = broad; n = narrow",
    "k = black; n = brown; b = buff; h = chocolate; g = gray; r = green; o = orange; p = pink; u = purple; e = red; w = white; y = yellow",
    "e = enlarging; t = tapering",
    "b = bulbous; c = club; u = cup; e = equal; z = rhizomorphs; r = rooted; ? = missing",
    "f = fibrous; y = scaly; k = silky; s = smooth",
    "f = fibrous; y = scaly; k = silky; s = smooth",
    "n = brown; b = buff; c = cinnamon; g = gray; o = orange; p = pink; e = red; w = white; y = yellow",
    "n = brown; b = buff; c = cinnamon; g = gray; o = orange; p = pink; e = red; w = white; y = yellow",
    "p = partial; u = universal",
    "n = brown; o = orange; w = white; y = yellow",
    "n = none; o = one; t = two",
    "c = cobwebby; e = evanescent; f = flaring; l = large; n = none; p = pendant; s = sheathing; z = zone",
    "k = black; n = brown; b = buff; h = chocolate; g = gray; r = green; o = orange; p = pink; u = purple; e = red; w = white; y = yellow",
    "a = abundant; c = clustered; n = numerous; s = scattered; v = several; y = solitary",
    "g = grasses; l = leaves; m = meadows; p = paths; u = urban; w = waste; d = woods"
  ),
  
  stringsAsFactors = FALSE
)

# ---------------------------
# Earthy color palette (expanded)
# ---------------------------
earthyPalette <- c("#556B2F","#8F9779","#6B8E23","#A0522D","#C2B280","#D2B48C","#7B3F00","#3B5323")
names(earthyPalette) <- NULL

# ---------------------------
# UI
# ---------------------------
ui <- dashboardPage(
  skin = "yellow",
  dashboardHeader(title = "🍄 Mushroom Classification Dashboard", titleWidth = 320),
  dashboardSidebar(width = 320,
                   # add id so we can detect active tab in server
                   sidebarMenu(id = "tabs",
                               menuItem("Home", tabName="home", icon=icon("home")),
                               menuItem("Dataset", tabName="data", icon=icon("table")),
                               menuItem("Penjelasan Dataset", tabName="doc", icon=icon("book")),
                               menuItem("Visualisasi & EDA", tabName="eda", icon=icon("chart-bar")),
                               menuItem("Feature & Model", tabName="feat_model", icon=icon("chart-line")),
                               menuItem("Prediction", tabName="pred", icon=icon("magic")),
                               menuItem("Author", tabName="author", icon=icon("users"))
                   )
  ),
  dashboardBody(
    tags$style(HTML(sprintf("\n      body, .content-wrapper { background: %s !important; }\n      .box { border-top: 4px solid %s !important; background: white; }\n      .box-header { background: %s !important; }\n      h1,h2,h3,h4,strong { color: %s !important; }\n      details summary { padding: 10px; background: %s; border-radius: 8px; font-size: 15px; cursor: pointer; font-weight:600; }\n      details summary:hover { background: %s; }\n      details div { background: #fff; padding: 12px; border-left: 4px solid %s; border-radius: 8px; margin-bottom: 10px; }\n      .muted { color: #6b6b6b; font-size: 13px; }\n    ", "#F6F2E9", "#3D2B1F", "#f0ede3", "#3D2B1F", "#8F9779", "#C2B280", "#2E4D3A"))),
    
    tabItems(
      # HOME
      tabItem(
        tabName = "home",
        
        fluidRow(
          box(
            width = 12, solidHeader = FALSE, status = "primary",
            
            div(style="padding:30px 20px;",
                
                # ========================== Row Atas: Gambar + Judul ==========================
                fluidRow(
                  # ---- Gambar kiri ----
                  column(
                    width = 4, align = "center",
                    tags$img(
                      src = "https://storage.googleapis.com/kaggle-datasets-images/478/974/557711140aeab7ca242d1e903c4e058e/dataset-cover.jpg",
                      style="
                      border-radius:16px;
                      box-shadow:0 8px 20px rgba(0,0,0,0.15);
                      width:100%;
                      height:220px;
                      object-fit:cover;
                    "
                    )
                  ),
                  
                  # ---- Judul kanan ----
                  column(
                    width = 8,
                    tags$div(
                      style="padding-left:15px;",
                      
                      h1(
                        "Mushroom Classification Dashboard",
                        style="
                        font-weight:700; 
                        color:#333; 
                        margin-bottom:10px;
                        margin-top:0;
                      "
                      ),
                      
                      p(
                        "Dashboard interaktif untuk mengeksplorasi ciri morfologi jamur, memahami fitur terpenting dalam klasifikasi, serta memprediksi kategori jamur (edible vs poisonous) menggunakan model Random Forest.",
                        style="color:#555; font-size:16px; line-height:1.5;"
                      )
                    )
                  )
                ),
                
                br(),
                
                # ========================== Row Info Dataset ==========================
                fluidRow(
                  column(width = 6, uiOutput("total_rows_card")),
                  column(width = 6, uiOutput("total_features_card"))
                ),
                
                br(),
                
                # ========================== Deskripsi Dataset ==========================
                div(
                  style="
                  margin-top:10px;
                  background:white;
                  border-radius:12px;
                  padding:24px;
                  box-shadow:0 4px 12px rgba(0,0,0,0.08);
                  color:#444;
                  line-height:1.65;
                  text-align:justify;
                ",
                  
                  fluidRow(
                    
                    # ================= LEFT SIDE - TEXT =================
                    column(
                      width = 8,
                      
                      p(
                        "Dataset ini berisi 8124 sampel jamur dengan 22 fitur morfologi yang diambil dari 23 spesies jamur berinsang di keluarga ",
                        tags$b("Agaricus"), " dan ", tags$b("Lepiota"),
                        ". Dataset ini awalnya berasal dari panduan ",
                        tags$i("The Audubon Society Field Guide to North American Mushrooms (1981).")
                      ),
                      
                      p(
                        "Setiap sampel diklasifikasikan sebagai jamur yang dapat dikonsumsi (edible) atau beracun (poisonous). Kategori ‘unknown edibility’ dalam buku asli digabungkan ke kelas beracun karena tidak direkomendasi untuk dikonsumsi."
                      ),
                      
                      p(
                        "Panduan aslinya menekankan bahwa tidak ada aturan sederhana untuk menentukan apakah jamur aman dikonsumsi. Karenanya, dataset ini sering digunakan untuk mengevaluasi model pembelajaran machine learning, serta memahami fitur morfologi apa yang paling menentukan sifat toksisitas jamur."
                      ),
                      
                      p(
                        "Di tengah meningkatnya tren mencari jamur liar ('shrooming') sebagai hobi di alam terbuka, kesalahan identifikasi jamur dapat berakibat fatal. Karena tidak ada aturan sederhana untuk membedakan jamur aman dan beracun hanya dari penampilan luar, dashboard ini menyediakan sarana eksplorasi data untuk memahami ciri morfologi yang paling berpengaruh dalam klasifikasi. Melalui visualisasi interaktif dan model machine learning, pengguna dapat melihat pola nyata dalam data, meningkatkan pemahaman risiko toksisitas jamur, serta mendukung edukasi berbasis data."
                      )
                    ),
                    
                    # ================= RIGHT SIDE - VIDEO =================
                    column(
                      width = 4,
                      div(
                        style="text-align:center;",
                        tags$iframe(
                          width="100%",
                          height="260",
                          src="https://www.youtube.com/embed/jbN0YBljPaM",
                          title="Shrooming Video",
                          frameborder="0",
                          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture",
                          allowfullscreen=NA
                        ),
                        tags$small(
                          "Sumber video: YouTube Shorts",
                          style="color:#777; display:block; margin-top:6px;"
                        )
                      )
                    )
                  )
                )
            ) # end div
          ) # end box
        ) # end fluidRow
      ), # end tabItem HOME
      
      # DATASET preview
      tabItem(tabName = "data",
              fluidRow(
                box(width = 12, title = "Preview Dataset", status="primary", solidHeader = TRUE,
                    p("Preview dataset (gunakan fitur pencarian atau scroll horizontal jika diperlukan).", class="muted"),
                    DTOutput("table_preview")
                )
              )
      ),
      
      # DOC - explanation accordion + table
      tabItem(
        tabName = "doc",
        fluidRow(
          column(
            width = 10, offset = 1,   # ← supaya tidak mepet kiri
            box(
              width = 12,
              title = "Penjelasan Dataset",
              status = "primary",
              solidHeader = TRUE,
              
              # ------------------- Accordion -------------------
              tags$details(
                style = "margin-bottom:20px; font-size:16px;",
                tags$summary("Penjelasan lengkap dataset (klik untuk membuka)", style="cursor:pointer; font-weight:600;"),
                
                tags$div(
                  style = "
              background:#fafafa;
              padding:20px 25px;
              border-radius:10px;
              margin-top:10px;
              line-height:1.65;
              font-size:15px;
              color:#444;
              box-shadow:0 4px 12px rgba(0,0,0,0.06);
            ",
                  
                  p("Dataset jamur ini awalnya disumbangkan ke UCI Machine Learning Repository lebih dari tiga dekade lalu. Meski sudah lama, kegiatan mencari jamur liar, sering disebut 'shrooming', semakin populer."),
                  p("Dataset ini berisi ciri-ciri morfologi dari berbagai spesies jamur dan digunakan untuk mempelajari faktor apa saja yang membedakan jamur aman dikonsumsi dengan yang beracun."),
                  p("Data ini disusun berdasarkan 23 spesies jamur berinsang dari famili Agaricus dan Lepiota, diambil dari The Audubon Society Field Guide to North American Mushrooms (1981)."),
                  p("Kelas 'tidak diketahui' dalam panduan tersebut digabungkan dengan kategori beracun karena tidak direkomendasikan untuk dikonsumsi."),
                  p("Panduan aslinya menekankan bahwa tidak ada aturan sederhana untuk menentukan aman–tidaknya jamur, sehingga dataset ini menjadi bahan menarik untuk mengevaluasi model machine learning dan fitur-fitur mana yang paling berperan dalam mendeteksi jamur beracun.")
                )
              ),
              
              # ------------------- Tabel Ringkas -------------------
              h4("Tabel ringkas: Variabel | Tipe | Sampel nilai"),
              br(),
              DTOutput("vars_table")
            )
          )
        )
      ),

      
      # EDA & Visualisasi (digabung) - large panels
      tabItem(tabName = "eda",
              fluidRow(
                box(width = 12, title = "Visualisasi & EDA (Pilih Variabel untuk Visualisasi Lebih Lanjut)",
                    status="primary", solidHeader = TRUE,
                    
                    selectInput("eda_var", "Pilih variabel (semua kolom):",
                                choices = names(data), 
                                selected = names(data)[2]),
                    
                    p("Panel menampilkan: Distribusi, Variabel vs Class (dodge), Proporsi (fill), dan Cramér's V heatmap.",
                      class="muted"),
                    
                    plotOutput("edaPanels", height = "850px") %>% withSpinner(type = 6)
                )
              )
      ),

      
      # Feature importance + Model Performance (DIGABUNG)
      tabItem(
        tabName = "feat_model",
        fluidRow(
          box(
            width = 12, 
            title = "Feature Importance (MI & Chi-Square) dan Model Performance",
            status = "warning", 
            solidHeader = TRUE,
            
            splitLayout(
              cellWidths = c("50%", "50%"),
              plotOutput("miPlot", height = "420px"),
              plotOutput("chiPlot", height = "420px")
            ),
            
            br(),
            DTOutput("feat_table"),
            hr(),
            
            h4("Model Performance Summary"),
            DTOutput("model_table"),
            br(),
            
            # ====== Catatan tambahan ======
            tags$div(
              style = "background:#fafafa; border-left:4px solid #f0ad4e; padding:10px; margin-bottom:12px;",
              tags$b("Catatan:"),
              p("Skor 1.00 pada Random Forest dan XGBoost tidak selalu menandakan overfitting. Dataset jamur bersifat highly separable, karena atribut morfologi seperti odor, spore-print-color, dan gill-size memiliki distribusi yang sangat berbeda antara kelas edible dan poisonous. Model tree-based mampu menangkap pola ini secara sempurna, baik pada holdout maupun validasi silang. Logistic Regression sedikit lebih rendah karena pendekatan linier tidak sepenuhnya menangkap interaksi antar fitur kategorikal.")
            ),
            # ===============================
            
            h4("Mengapa memilih Random Forest?"),
            p(
              "Random Forest dipilih karena mencapai performa sempurna tanpa tuning kompleks, stabil pada validasi silang, dan bekerja secara natural dengan fitur kategorikal. Model ini memiliki hyperparameter lebih sedikit, lebih mudah direplikasi, dan analisis feature importance-nya lebih intuitif, sehingga pengguna dapat memahami faktor morfologi yang mempengaruhi klasifikasi jamur. Pada dataset jamur yang bersih dan mudah dipisahkan, kompleksitas ekstra XGBoost tidak memberikan manfaat tambahan yang berarti.",
              class = "muted"
            ),
            
            br(),
            h4("Random Forest Diagnostics"),
            p("Variable importance, OOB error, Confusion matrix, dan ROC di bawah ini.", class = "muted"),
            
            # ==== Baris 1: VarImp + OOB ====
            div(
              style = "
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 25px;
          margin-bottom: 25px;
        ",
              
              plotOutput("rf_varimp", height = "280px"),
              plotOutput("rf_oob", height = "280px")
            ),
            
            # ==== Baris 2: Confusion Matrix + ROC ====
            div(
              style = "
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 25px;
        ",
              
              DTOutput("rf_conf_table"),
              plotOutput("rf_roc", height = "340px")
            )
            
          ) # box
        ) # fluidRow
      ), # tabItem
      
      
      # Prediction (own tab)
      tabItem(tabName = "pred",
              fluidRow(
                box(width = 6, title = strong("Input Mushroom Attributes (Random Forest)"), status="primary", solidHeader = TRUE,
                    p("Pilih nilai untuk setiap fitur (pilihan dari data atau mapping). Mapping otomatis ke kode internal.", class="muted"),
                    uiOutput("predict_ui"),
                    actionButton("btn_predict", "Predict!", class="btn btn-success", style="margin-top:10px;")
                ),
                box(width = 6, title = "Result & Info", status="info", solidHeader = TRUE,
                    withSpinner(textOutput("predict_result"), type = 6),
                    br(),
                    tags$details(tags$summary("Catatan"), tags$div("Prediksi ini berdasarkan pola data historis dan tidak menggantikan evaluasi lapangan oleh ahli mikologi. Model hanya memberikan indikasi probabilistik dan tidak boleh digunakan sebagai dasar tunggal dalam pengambilan keputusan."))
                )
              )
      ),
        tabItem(
          tabName = "author",
          
          fluidRow(
            box(
              width = 12,
              title = "Kelompok 16 - Author",
              status = "primary",
              solidHeader = TRUE,
              
              # =============================
              # FOTO UTAMA (HEADER)
              # =============================
              div(
                style = "
          width: 100%;
          display: flex;
          justify-content: center;
          padding: 20px 0;
        ",
                
                tags$img(
                  src = "https://i.pinimg.com/1200x/ef/60/42/ef604244da8284eea9ddefaa7ca776ca.jpg",
                  style = "
            width: 100%;
            max-width: 1100px;
            height: 350px;
            border-radius: 25px;
            object-fit: cover;
            box-shadow: 0 4px 14px rgba(0,0,0,0.25);
          "
                )
              ),
              
              # =============================
              # BAGIAN NAMA AUTHOR
              # =============================
              div(
                style = "
          text-align: center;
          padding: 10px 25px 25px 25px;
        ",
                
                h3(
                  "Anggota Kelompok 16",
                  style = "font-weight:600; margin-bottom:15px;"
                ),
                
                tags$ul(
                  style = "
            list-style: none;
            padding: 0;
            font-size: 17px;
            line-height: 1.6;
          ",
                  tags$li("🍄 Auliya Nasywa Hafiyyanti  
                    🍄 Shafa Insan Kamillia Ashari  
                    🍄 Oryza Sativa")
                ),
                
                p(
                  "Terima kasih telah menggunakan Mushroom Forest Dashboard.
           Dashboard ini dibuat sebagai bagian dari pemenuhan Final Project
           mata kuliah Data Mining dan Visualisasi.",
                  style = "
            font-size: 16px;
            line-height: 1.5;
            color: #555;
            max-width: 700px;
            margin: 15px auto 0;
          "
                )
              )
            )
          )
        )
        
      )
      
      
      
      
    
  )
)

# ---------------------------
# SERVER
# ---------------------------
server <- function(input, output, session){
  
  output$total_rows_card <- renderUI({
    div(
      style="
      background:#fafafa;
      padding:20px;
      border-radius:10px;
      text-align:center;
      box-shadow:0 3px 8px rgba(0,0,0,0.05);
    ",
      
      h3(style="margin-bottom: 6px; font-weight:700; color:#333;",
         format(nrow(data), big.mark=",")
      ),
      p(style="color:#777; margin:0;",
        "Total Observasi"
      )
    )
  })
  
  output$total_features_card <- renderUI({
    div(
      style="
      background:#fafafa;
      padding:20px;
      border-radius:10px;
      text-align:center;
      box-shadow:0 3px 8px rgba(0,0,0,0.05);
    ",
      
      h3(style="margin-bottom: 6px; font-weight:700; color:#333;",
         ncol(data)-1
      ),
      p(style="color:#777; margin:0;",
        "Total Fitur"
      )
    )
  })
  
  # Dataset preview
  output$table_preview <- renderDT({
    req(input$tabs == "data")
    datatable(data, options = list(pageLength = 10, scrollX = TRUE))
  })
  
  # Vars table (Variable | Type | Unique sample) - show explanation table on doc tab
  vars_info <- reactive({
    tibble::tibble(
      Variable = names(data),
      Type = sapply(data, function(x) class(x)[1]),
      Sample_Values = sapply(data, function(x){
        vals <- unique(as.character(x))
        if(length(vals) > 8) paste0(paste0(vals[1:7], collapse = ", "), ", ...") else paste0(vals, collapse = ", ")
      })
    )
  })
  # Only expose the structured variable explanation on the doc tab (remove duplicate assignment)
  output$vars_table <- renderDT({
    req(input$tabs == "doc")
    datatable(vars_desc, options = list(pageLength = 10, scrollX = TRUE))
  })
  
  # ---------------------------
  # EDA panels (NO stacked plot)
  # ---------------------------
  output$edaPanels <- renderPlot({
    req(input$tabs == "eda")
    req(input$eda_var)
    var <- input$eda_var
    if(! var %in% names(data)) return(NULL)
    dfv <- data %>% select(all_of(c(var,"Class"))) %>% drop_na()
    
    col_edible <- earthyPalette[2]
    col_poison <- earthyPalette[4]
    
    # DISTRIBUSI
    p1 <- ggplot(dfv, aes(x = !!sym(var))) +
      geom_bar(fill = col_edible) +
      theme_minimal(base_size = 14) +
      labs(title = paste("Distribusi:", var), x = var, y = "Count") +
      theme(plot.title = element_text(face = "bold"))
    
    # VAR vs CLASS (dodge)
    p2 <- ggplot(dfv, aes(x = !!sym(var), fill = Class)) +
      geom_bar(position = "dodge") +
      scale_fill_manual(values = c("edible" = col_edible, "poisonous" = col_poison)) +
      theme_minimal(base_size = 14) +
      labs(title = paste(var, "vs Class (count)"), x = var, y = "Count")
    
    # PROPORSI (fill)
    p3 <- ggplot(dfv, aes(x = !!sym(var), fill = Class)) +
      geom_bar(position = "fill") +
      scale_fill_manual(values = c("edible" = col_edible, "poisonous" = col_poison)) +
      theme_minimal(base_size = 14) +
      labs(title = paste("Proporsi Class per", var), x = var, y = "Proportion")
    
    # CRAMER'S V
    cat_cols <- names(data)[sapply(data, is.factor)]
    cramersV <- function(x, y){
      tbl <- table(x, y)
      chi2 <- suppressWarnings(chisq.test(tbl, simulate.p.value = FALSE)$statistic)
      n <- sum(tbl)
      k <- min(nrow(tbl), ncol(tbl))
      sqrt(chi2 / (n * (k - 1)))
    }
    v_scores <- sapply(cat_cols, function(cn){
      if(cn == var) return(NA)
      cramersV(dfv[[var]], data[[cn]])
    })
    vdf <- data.frame(var2 = cat_cols, V = v_scores, stringsAsFactors = FALSE) %>%
      filter(!is.na(V)) %>% arrange(desc(V))
    vdf_top <- head(vdf, 8)
    
    p4 <- ggplot(vdf_top, aes(x = reorder(var2, V), y = V, fill = V)) +
      geom_col() +
      coord_flip() +
      scale_fill_gradient(low = "#f0e9df", high = earthyPalette[4]) +
      theme_minimal(base_size = 13) +
      labs(title = paste("Cramér's V:", var, "vs other categorical vars"),
           x = "Variable", y = "Cramér's V")
    
    # Gabungkan semua panel (sekarang 4 plot saja)
    layout <- (p1 / p2 / p3 / p4) + plot_layout(ncol = 1, heights = rep(1,4))
    layout
  })
  
  
  # Feature importance plots & table
  # ---------------------------
  # MI plot (all 22 features)
  output$miPlot <- renderPlot({
    req(input$tabs == "feat_model")
    fr <- full_feature_rank
    fr$Feature <- factor(fr$Feature, levels = fr$Feature[order(fr$MI)])
    ggplot(fr, aes(x = MI, y = Feature)) +
      geom_col(fill = earthyPalette[2]) +
      theme_minimal(base_size = 13) +
      labs(title = "Mutual Information (MI) - Semua Fitur", x = "MI Score", y = "Feature")
  })
  
  # Chi-square plot (all 22 features) - move NA (e.g. veil-type) to bottom
  output$chiPlot <- renderPlot({
    req(input$tabs == "feat_model")
    fr <- full_feature_rank
    # create ordering: non-NA first by ChiSquare asc, then NA last
    fr <- fr %>% mutate(is_na = is.na(ChiSquare)) %>% arrange(is_na, ChiSquare)
    # for plotting, replace NA with 0 so the bar is minimal
    fr$ChiPlotVal <- ifelse(is.na(fr$ChiSquare), 0, fr$ChiSquare)
    fr$Feature <- factor(fr$Feature, levels = fr$Feature[order(fr$ChiPlotVal)])
    ggplot(fr, aes(x = ChiPlotVal, y = Feature)) +
      geom_col(fill = earthyPalette[4]) +
      theme_minimal(base_size = 13) +
      labs(title = "Chi-Square Score - Semua Fitur", x = "Chi-Square", y = "Feature")
  })
  
  # Feature table: MI + Chi, filter top12 explanation
  output$feat_table <- renderDT({
    req(input$tabs == "feat_model")
    top12 <- full_feature_rank %>% arrange(desc(MI)) %>% head(12)
    datatable(top12, options = list(pageLength = 12, scrollX = TRUE))
  })
  
  
  # Model performance table
  output$model_table <- renderDT({
    req(input$tabs == "feat_model")
    datatable(model_results, options = list(pageLength = 5, searching = FALSE, paging = FALSE), rownames = FALSE)
  })
  
  # Random Forest diagnostics
  output$rf_varimp <- renderPlot({
    req(input$tabs == "feat_model")
    vi <- importance(rf_model)
    vi_df <- data.frame(Feature = rownames(vi), MeanDecreaseGini = vi[, "MeanDecreaseGini"], stringsAsFactors = FALSE)
    vi_df <- vi_df %>% arrange(MeanDecreaseGini)
    ggplot(vi_df, aes(x = MeanDecreaseGini, y = reorder(Feature, MeanDecreaseGini))) +
      geom_col(fill = earthyPalette[1]) +
      theme_minimal(base_size = 13) +
      labs(title = "Random Forest - Variable Importance (MeanDecreaseGini)", x = "Importance", y = "Feature")
  })
  output$rf_oob <- renderPlot({
    req(input$tabs == "feat_model")
    plot(rf_model, main = "Random Forest OOB Error Rate")
  })
  # Confusion matrix as table
  output$rf_conf_table <- renderDT({
    req(input$tabs == "feat_model")
    pred_train <- predict(rf_model, data)
    cm <- table(True = data$Class, Pred = pred_train)
    # normalized
    cm_norm <- prop.table(cm, 1)
    # render both
    cm_df <- as.data.frame.matrix(cm)
    cm_df_norm <- round(as.data.frame.matrix(cm_norm), 3)
    # show original counts and normalized below as combined display via HTML
    html <- "<h4>Counts</h4>"
    html <- paste0(html, DT::datatable(cm_df, options = list(dom = 't')) %>% as.character())
    html <- paste0(html, "<br/><h4>Row-normalized (proportion)</h4>")
    html <- paste0(html, DT::datatable(cm_df_norm, options = list(dom = 't')) %>% as.character())
    # show as HTML in UI via datatable wrapper - easier: return cm_df (counts) and let UI see it
    datatable(cm_df, options = list(dom='t'), caption = htmltools::tags$caption(style = 'caption-side: bottom; text-align:left;','Above: Counts; Below: Row-normalized proportions shown separately.'))
  })
  # ROC plot
  output$rf_roc <- renderPlot({
    req(input$tabs == "feat_model")
    probs <- tryCatch({ predict(rf_model, data, type = "prob")[, "edible"] }, error = function(e) NULL)
    if(is.null(probs)) {
      plot.new(); text(0.5,0.5,"Probabilities unavailable for ROC plot.")
      return()
    }
    roc_obj <- roc(response = data$Class, predictor = probs, levels = c("poisonous","edible"), direction = "<")
    plot(roc_obj, main = paste0("ROC Random Forest (AUC = ", round(auc(roc_obj),3), ")"), col = earthyPalette[1])
  })
  
  # Prediction UI inputs
  output$predict_ui <- renderUI({
    req(input$tabs == "pred")
    lapply(available_features, function(fkey){
      colname <- feature_col_map[[fkey]]
      inputId <- paste0("in_", gsub("[^A-Za-z0-9]", "_", fkey))
      # present human-readable choices if cat_map exists, otherwise data unique values
      if(!is.null(cat_map[[fkey]])){
        choices <- names(cat_map[[fkey]])
      } else {
        choices <- sort(unique(as.character(data[[colname]])))
      }
      selectInput(inputId, label = gsub("\\.", " ", fkey), choices = choices, selected = choices[1])
    })
  })
  
  # Handle prediction
  observeEvent(input$btn_predict, {
    # allow predict event to run anytime, but only show result on pred tab
    newvals <- sapply(available_features, function(fkey){
      inid <- paste0("in_", gsub("[^A-Za-z0-9]", "_", fkey))
      sel <- input[[inid]]
      if(!is.null(cat_map[[fkey]]) && sel %in% names(cat_map[[fkey]])) {
        return(cat_map[[fkey]][[sel]])
      } else return(as.character(sel))
    }, USE.NAMES = TRUE)
    newrow <- as.data.frame(as.list(newvals), stringsAsFactors = FALSE)
    names(newrow) <- unname(unlist(feature_col_map[available_features]))
    # convert factors
    for(col in names(newrow)){
      if(col %in% names(data)){
        newrow[[col]] <- factor(newrow[[col]], levels = levels(data[[col]]))
      } else newrow[[col]] <- factor(newrow[[col]])
    }
    pred <- tryCatch(predict(rf_model, newrow), error = function(e) NA)
    output$predict_result <- renderText({
      req(input$tabs == "pred")
      if(is.na(pred)) return("Prediction error — periksa input / mapping.")
      if(as.character(pred) == "edible") paste0("🍄 Prediction: ", as.character(pred)) else paste0("⚠ Prediction: ", as.character(pred))
    })
  })
}

# Run app
shinyApp(ui, server)


