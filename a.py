import os
import pandas as pd
import google.generativeai as genai
from google.generativeai.types import generation_types
from dotenv import load_dotenv
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Load environment variables
load_dotenv()

# Configure the API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

data=f"""You are an expert in analyzing CSV dataset related to agriculture, 
    the format of the csv data is (entity1,head_label,relation,entity2,tail_label).
    the data is given below:
    entity1,head_label,relation,entity2,tail_label
    Agricultural,Agri_Process,Conjunction,horticultural,Agri_Process
    yellow stem bore,Organism,Conjunction,rice leaf folder,Organism
    millet crops,Crop,Includes,brown top millet,Crop
    Agricultural contaminants,Agri_Pollution,Includes,pesticides,Chemical
    storms,Natural_Disaster,Conjunction,flood,Natural_Disaster
    yield-enhancing,O,Conjunction,variance-reducing,O
    Multi Layer Perceptron,ML_Model,Synonym_Of,MLP,ML_Model
    staple food crops,Crop,Includes,corn,Crop
    fungi,Organism,Includes,Fusarium,Organism
    The hydroponic technique,Agri_Method,Used_For,green fodder,Crop
    hydroponics fodder,Crop,Used_For,animals,Organism
    Fusarium wilt,Disease,Caused_By,Fusarium oxysporum,Organism
    agronomic crops,Crop,Includes,roots and tuber crops,Crop
    pear,Fruit,Conjunction,apple,Fruit
    CTR,Agri_Method,Synonym_Of,conventional transplanted rice,Agri_Method
    Indo-Gangetic Plains of India,Organization,Synonym_Of,IGP,Organization
    South Eastern region,Location,Includes,Montenegro,Location
    bacteria,Organism,Conjunction,fungi,Organism
    oil palm,Crop,Origin_Of,West Africa,Location
    NRC,Organization,Synonym_Of,National Research Council,Organization
    RMSE,ML_Model,Synonym_Of,Root mean square error,ML_Model
    shrub lands,Field_Area,Conjunction,fallows,Field_Area
    millets,Crop,Includes,Fonio,Crop
    Mango,Fruit,Includes,Chausa mango,Fruit
    measurement of genetic diversity,Other_Quantity,Helps_In,crop improvement programme,Agri_Method
    agroforestry models,Agri_Process,Includes,Agri-Silviculture,Agri_Process
    agroforestry,Agri_Process,Helps_In,microclimate,Agri_Process
    Digitaria exilis,Crop,Synonym_Of,Fonio,Crop
    soil moisture fluctuation,Natural_Disaster,Coreference,soil moisture,Natural_Disaster
    sediments,Agri_Pollution,Caused_By,industrial activities,Other
    soil microorganism,Organism,Helps_In,soil health,Soil
    Xoo,Location,Synonym_Of,Xanthomonas oryzae pathovar oryzae,Organism
    North Eastern region,Location,Includes,Czech Republic,Location
    fungi,Organism,Conjunction,soil-borne bacteria,Organism
    sorghum,Crop,Synonym_Of,Sorghum bicolor,Crop
    Agri-Horti-Silviculture,Agri_Process,Helps_In,crops,Crop
    PATs,Technology,Synonym_Of,precision agricultural technologies,Technology
    yellow stem borer,Organism,Conjunction,rice leaf folder,Organism
    nutrients,Nutrient,Includes,K,Nutrient
    South Eastern region,Location,Includes,Serbia,Location
    organic matter,Natural_Resource,Includes,animal wastes,Agri_Waste
    oomycetes,Organism,Includes,Phytophthora,Organism
    rainfed dryland crops,Crop,Coreference,rainfed cropping,Crop
    millet crops,Crop,Includes,pearl millet,Crop
    nutrients,Nutrient,Includes,B,Nutrient
    dessert,Food_Item,Includes,muffins,Food_Item
    agroforestry,Agri_Process,Helps_In,soil moisture,Soil
    conventional transplanted rice,Agri_Method,Synonym_Of,CTR,Agri_Method
    bacterial blight,Disease,Origin_Of,Caribbean,Location
    agronomic crops,Crop,Includes,garden crops,crop
    Root rot,Disease,Caused_By,Aphanomyces,Organism
    Alphonso,Fruit,Conjunction,Kesar mangoes,Fruit
    foliar fungal diseases,Disease,Includes,downy mildew,Disease
    apple,Fruit,Origin_Of,North America,Location
    Government of India,Organization,Coreference,World Bank,Organization
    greenhouse gases,Chemical,Includes,carbon dioxide,Chemical
    bird species,Organism,Coreference,Lesser Grey Shrike,Organism
    P,Nutrient,Synonym_Of,Phosphorus,Nutrient
    Natural disasters,Natural_Disaster,Includes,hydro-meteorological,Natural_Disaster
    AgriHorticulture,Agri_Process,Helps_In,crops,Crop
    GHG,Chemical,Synonym_Of,greenhouse gases,Chemical
    Bacterial wilt,Disease,Caused_By,Erwinia,Organism
    CO2,Chemical,Caused_By,deforestation,Other
    agronomic crops,Crop,Includes,vegetable,Vegetable
    Rot,Disease,Caused_By,funguslike organisms,Organism
    horticultural,Agri_Process,Conjunction,Agricultural,Agri_Process
    millets,Crop,Includes,Job’s tears,Crop
    Shade trees,Crop,Includes,maple,Crop
    south-western coastal,Location,Includes,Karnataka,Location
    food crops,Crop,Includes,melons,Fruit
    contaminating the soils,Agri_Pollution,Caused_By,industrial activities,Other
    Senior et al 2013,Citation,Conjunction,Edwards et al 2013,Citation
    CO2,Chemical,Synonym_Of,carbon dioxide,Chemical
    maize grain,Crop,Conjunction,sugar beet,Crop
    Kesar mangoes,Fruit,Conjunction,Alphonso,Fruit
    fodder crops,Crop,Includes,Mustard,Crop
    phosphorus,Nutrient,Synonym_Of,P,Nutrient
    finger millet,Crop,Includes,ragi,Crop
    Xanthomonas oryzae pathovar oryzae,Organism,Synonym_Of,Xoo,Organism
    Pradhan Mantri Krishi Sinchayee Yojana,Policy,Synonym_Of,PMKSY,Policy
    Jackfruit,Fruit,Origin_Of,India,Location
    green grapes,Fruit,Coreference,grapes,Fruit
    climate change,Natural_Disaster,Includes,global warming,Natural_Disaster
    low-cost protected technologies,Technology,Includes,plastic low tunnels,Technology
    North Eastern region,Location,Includes,Poland,Location
    Asia,Location,Origin_Of,tropical forests,Location
    millet crops,Crop,Includes,proso millet,Crop
    agricultural land,Field_Area,Includes,shrublands,Field_Area
    greenhouse gases,Chemical,Includes,methane,Chemical
    nitrogen,Nutrient,Synonym_Of,N,Nutrient
    vegetable transplanting mechanism,Technology,Includes,automatic mechanism,Technology
    kharif seasons,Season,Includes,rainy,Season
    Heavy metals,Natural_Resource,Includes,Pb,Chemical
    NDVI,ML_Model,Synonym_Of,Normalized Difference Vegetation Index,ML_Model
    Basal rot,Disease,Caused_By,bacteria,Organism
    Fonio,Crop,Synonym_Of,Digitaria exilis,Crop
    nutrients,Nutrient,Includes,N,Nutrient
    Agricultural contaminants,Agri_Pollution,Includes,herbicides,Chemical
    wilt diseases,Disease,Caused_By,bacteria,Organism
    forage crop,Crop,Includes,ryegrass,Crop
    AR,Crop,Synonym_Of,aerobic rice,Crop
    Soil organic carbon,Chemical,Synonym_Of,SOC,Chemical
    kharif,Crop,Seasonal,rainy,Season
    organic farming and tillage,Agri_Method,Helps_In,soil health,Soil
    hay,Agri_Waste,Conjunction,silage,Agri_Waste
    acervuli,Disease,Caused_By,Colletotrichum,Organism
    apple,Fruit,Origin_Of,Kashmir ,Location
    Root rot,Disease,Caused_By,fungi,Organism
    Alphonso,Fruit,Origin_Of,India,Location
    oil palm,Crop,Origin_Of,tropical forest,Location
    Training and Visit (T&V) extension,Agri_Method,Conjunction,Green Revolution technology,Technology
    Brazil,Location,Includes,Foz do Iguaçu,Location
    soil-borne bacteria,Organism,Conjunction,fungi,Organism
    healthy soil,Soil,Helps_In,controlling soil nutrient recycling decomposition,Agri_Method
    pearl millet,Crop,Includes,bajra,Location
    under-nutrition,Disease,Conjunction,malnutrition,Disease
    crop diversification,Agri_Process,Helps_In,weeds and soil erosion,Natural_Disaster
    mean temperature,Weather,Caused_By,climate change,Natural_Disaster
    Summer,Season,Conjunction,Spring,Season
    South Eastern region,Location,Includes,Herzegovina,Location
    oaks,Location,Origin_Of,European,Location
    Root rot,Disease,Caused_By,Pythium,Organism
    Agricultural contaminants,Agri_Pollution,Includes,pesticides,Chemical
    agronomic crops,Crop,Includes,oil seed crop,Crop
    Gray mold rot,Disease,Caused_By,genus Botrytis,Organism
    Normalized Difference Vegetation Index,ML_Model,Synonym_Of,NDVI,ML_Model
    millet crops,Crop,Includes,sorghum,Crop
    Mango,Fruit,Includes,Safeda,Fruit
    tropical forest,Location,Coreference,128 tropical countries,Other_Quantity
    seven treatments,Treatment,Includes,Soil moisture-based drip irrigation to 100% FC,Agri_Method
    EPIC,ML_Model,Synonym_Of,Erosion/Productivity Impact Calculato,ML_Model
    P,Nutrient,Helps_In,crop production,Agri_Process
    agronomic crops,Crop,Includes,cereal,Crop
    oranges,Fruit,Seasonal,autumn,Season
    bulb rot,Disease,Conjunction,Basal rot,Disease
    MLP,ML_Model,Synonym_Of,Multi Layer Perceptron,ML_Model
    Sudan Savannah,Location,Includes,north-eastern part of Upper East region,Location
    Job’s tears,Crop,Synonym_Of,Coix lacryma-jobi,Crop
    cereal crop,Crop,Origin_Of,Assam,Location
    fodder crops,Crop,Includes,Oat/Barley,Crop
    greenhouse gases,Chemical,Includes,nitrous oxide,Chemical
    Bulgaria,Location,Conjunction,Romania,Location
    staple food crops,Crop,Includes,wheat,Crop
    hydroponic fodder,Crop,Used_For,livestock,Organism
    Setaria italica,Crop,Includes,proso millet,Crop
    south-western coastal,Location,Includes,Tamilnadu,Location
    Agricultural contaminants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    agricultural land,Field_Area,Includes,savannas,Field_Area
    Nitrogen,Nutrient,Used_For,agricultural cropping systems,Agri_Process
    agronomic crops,Crop,Includes,forage crops,Crop
    nitrogen,Nutrient,Helps_In,crop yield,Agri_Process
    contaminating the soils,Agri_Pollution,Conjunction,sediments,Agri_Pollution
    green fodder,Crop,Used_For,dairy animals,Organism
    barnyard millet,Crop,Synonym_Of,Echinochloa frumentacea,Crop
    Root rot,Disease,Caused_By,Fusarium,Organism
    Arbuscular mycorrhizal fungi,Organism,Synonym_Of,AMF,Organization
    hydrometeorological disasters,Natural_Disaster,Includes,forest fires,Natural_Disaster
    Basal rot,Disease,Caused_By,fungi,Organism
    millet crops,Crop,Includes,barnyard millet,Crop
    nitrogen,Nutrient,Conjunction,phosphorus,Nutrient
    nitrogen,Nutrient,Helps_In,pulses,Crop
    Fig,Fruit,Seasonal,summers,Season
    GCMs,ML_Model,Synonym_Of,global climate models,ML_Model
    AKIS,Organization,Synonym_Of,agricultural knowledge and innovation systems,Organization
    rainfed cropping,Crop,Coreference,rainfed dryland crops,Crop
    Logistic Model Trees,ML_Model,Synonym_Of,LMT,ML_Model
    South Eastern region,Location,Includes,Slovenia,Location
    Bacterial wilt,Disease,Caused_By,Pseudomonas,Organism
    south-western coastal,Location,Includes,Gujarat,Location
    Oak wilt,Crop,Caused_By,fungus Ceratocystis fagacearum,Organism
    wilt diseases,Disease,Caused_By,fungi,Organism
    little millet,Crop,Includes,kutki,Crop
    2070,Date_and_Time,Conjunction,2050,Date_and_Time
    old farming areas,Field_Area,Helps_In,Yield improvement,Agri_Process
    bacterial blight,Disease,Origin_Of,Australia,Location
    Green Revolution technology,Technology,Origin_Of,India,Location
    proso millet,Crop,Includes,cheena,Crop
    Eragrostis tef,Crop,Synonym_Of,Teff,Crop
    seven treatments,Treatment,Includes,Timer based drip irrigation to 100% CWR,Agri_Method
    droughts,Natural_Disaster,Conjunction,storms,Natural_Disaster
    optimizing fertilizer timing and placement,Treatment,Helps_In,cropping systems,Agri_Process
    National Research Council,Organization,Origin_Of,U.S.,Location
    fungi,Organism,Conjunction,bacteria,Organism
    Telangana State,Location,Includes,Ranga Reddy,Location
    farm,Field_Area,Conjunction,orchard,Field_Area
    dogwood anthracnose,Disease,Origin_Of,North America,Location
    healthy soil,Soil,Helps_In,plant productivity,Agri_Process
    Silvi-Pasture,Crop,Includes,fodder crops,Crop
    2019,Date_and_Time,Conjunction,2018,Date_and_Time
    PMKSY,Policy,Synonym_Of,Pradhan Mantri Krishi Sinchayee Yojana,Policy
    Shade trees,Crop,Includes,sycamore,Crop
    Agricultural contaminants,Agri_Pollution,Includes,insecticides,Chemical
    food crops,Crop,Includes,bananas,Fruit
    guavas,Fruit,Seasonal,early summers,Season
    Streptomycin,Chemical,Helps_In,Fire blight,Disease
    decision support tools,Technology,Helps_In,cropping systems,Agri_Process
    CO2,Chemical,Caused_By,fossil fuels,Natural_Resource
    Zea mays,Crop,Synonym_Of,maize,Crop
    yellow stem borer,Organism,Synonym_Of,Scirpophaga incertulas,Organism
    automation,Technology,Helps_In,vegetable cultivation,Agri_Process
    apple,Fruit,Coreference,green apple,Fruit
    Aster yellows,Disease,Caused_By,phytoplasma bacterium,Organism
    apple,Fruit,Conjunction,pear,Fruit
    Green Revolution technology,Technology,Conjunction,Training and Visit (T&V) extension,Agri_Method
    dessert,Food_Item,Includes,cakes,Food_Item
    seven treatments,Treatment,Includes,Timer based drip irrigation to 80% CWR,Agri_Method
    brown top millet,Crop,Synonym_Of,Brachiaria ramosa,Crop
    Upper Brahmaputra valley zones,Location,Includes,North Bank Plains,Location
    FAO,Organization,Synonym_Of,Food and Agriculture Organization,Organization
    putrefaction,Agri_Process,Conjunction,plant decomposition,O
    bacterial blight,Disease,Origin_Of,Asia,Location
    Heavy metals,Crop,Includes,Zn,O
    foliar fungal diseases,Disease,Includes,leaf yellowing,Disease
    fodder crops,Crop,Includes,Lucerne,Location
    Farmers,Person,Coreference,farmers,Person
    hydrometeorological disasters,Natural_Disaster,Includes,floods,Natural_Disaster
    2021,Date_and_Time,Conjunction,2022,Date_and_Time
    farmers,Person,Coreference,Farmers,Person
    food crop,Crop,Includes,Wheat,Crop
    Paspalum scrobiculatum,Crop,Synonym_Of,kodo millet,Crop
    millet crops,Crop,Includes,little millet,Crop
    watershed-level strategies,O,Helps_In,cropping systems,Agri_Process
    International Conference of Agricultural Economists,Organization,Origin_Of,Brazil,Location
    Bacterial wilt,Disease,Caused_By,Xanthomonas,Crop
    carbon dioxide,Chemical,Conjunction,methane,Nutrient
    farmer,Person,Conjunction,farmer,Person
    Watermelon,Fruit,Seasonal,summer,Season
    Ferterra,Location,Includes,Chlorantraniliprole,Chemical
    agronomic crops,Crop,Includes,pulses,crop
    Climate change,Natural_Disaster,Includes,rise in sea level,O
    MaxEnt,ML_Model,Synonym_Of,Maximum entropy,O
    millet crops,Crop,Includes,oxtail millet,Crop
    rabi,O,Seasonal,winter,Season
    Fire bligh,Disease,Caused_By,Erwinia amylovora,Crop
    dates,O,Origin_Of,Middle East,Location
    paddy straw,Agri_Waste,Conjunction,husk,Crop
    AIS,Location,Coreference,AIS,Location
    sugar beet,Crop,Conjunction,maize grain,Crop
    Agri-Silviculture,Agri_Process,Includes,crops,O
    millet crops,Crop,Includes,finger millet,Crop
    Mango,Fruit,Includes,Langra,Location
    cultivated rice,Crop,Includes,Oryza sativa,O
    Shade trees,Crop,Includes,oak,O
    rainfall,O,Conjunction,NDVI,Location
    India,Location,Includes,Telangana State,Location
    North Bank Plains,Location,Conjunction,Upper Brahmaputra valley zones,Location
    N,O,Used_For,cropping systems,Agri_Process
    NARS,Location,Origin_Of,sub-Saharan Africa,Location
    Agronomy,Agri_Process,Includes,staple crops,Crop
    Rhizobium radiobacter,Organism,Synonym_Of,Agrobacterium tumefaciens,Organism
    food crops,Crop,Includes,sweet potatoes,Vegetable
    Erosion/Productivity Impact Calculato,O,Synonym_Of,EPIC,Organization
    finger millet,Crop,Synonym_Of,Eleusine coracana,Crop
    Latin America,Location,Origin_Of,tropical forests,Location
    Chestnut-collared Longspur,Crop,Synonym_Of,Calcarius ornatus,Organism
    NARS,Location,Coreference,NARS,Location
    PBs,Crop,Synonym_Of,wheat on permanent beds,Crop
    water,Natural_Resource,Helps_In,ryegrass cultivation,Agri_Process
    Root rot,Disease,Caused_By,Clitocybe tabescens,Organism
    Soil health,Soil,Helps_In,sustainable agriculture,Agri_Process
    fodder crops,Crop,Includes,Berseem,Location
    Climate change,Natural_Disaster,Includes,increased temperatures,O
    acervuli,O,Caused_By,Gloeosporium,Organism
    University of Agricultural Sciences,Organization,Origin_Of,Karnataka,Location
    tree based,O,Includes,LMT,Technology
    Rosaceae,Organism,Synonym_Of,rose,O
    dessert,O,Includes,cupcakes,Food_Item
    foliar fungal diseases,Disease,Includes,anthracnose,Disease
    food crops,Crop,Includes,legumes,Crop
    yield-enhancing,O,Coreference,yield-enhancement,O
    Central Europe,Location,Synonym_Of,CE,Location
    Xoo,Location,Synonym_Of,Xanthomonas oryzae pathovar oryzae,Organism
    foliar fungal diseases,Disease,Includes,Ascochyta blight,Disease
    Root rot,Disease,Caused_By,oomycetes,Chemical
    south-western coastal,Location,Includes,Kerala,Location
    Haryana,Location,Includes,Hisar,Location
    Training and Visit (T&V) extension,Agri_Method,Origin_Of,India,Location
    South Eastern region,Location,Includes,Bosnia,Location
    SOC,O,Synonym_Of,Soil organic carbon,Chemical
    humid environments,Weather,Helps_In,bacterial blight,Disease
    Rot,O,Caused_By,soil-borne bacteria,Organism
    Europe,Location,Coreference,Europe,Location
    irrigation,Agri_Process,Conjunction,anti-transpirants,Chemical
    PATs,Other_Quantity,Synonym_Of,precision agricultural technologies,Technology
    agronomic crops,Crop,Includes,fibre crops,Crop
    Heavy metals,Crop,Includes,Fe,O
    Brachiaria ramosa,Crop,Synonym_Of,brown top millet,Crop
    apple,Fruit,Seasonal,winter,Season
    water,Natural_Resource,Coreference,resources,O
    NZ,Location,Synonym_Of,New Zealand,Location
    orchard,Field_Area,Conjunction,forest,O
    Basal rot,Disease,Synonym_Of,bulb rot,Disease
    ICAR-Indian Agricultural Research Institute,Organization,Origin_Of,New Delhi,Location
    south-western coastal,Location,Includes,Maharashtra,Location
    maize,Crop,Synonym_Of,Zea mays,Crop
    biomass harvesting,Agri_Process,Helps_In,tree survival and growt,O
    AMF,Organization,Synonym_Of,Arbuscular mycorrhizal fungi,Organism
    oomycetes,Chemical,Includes,Pythium,Location
    little millet,Crop,Includes,sama,O
    NARS,Location,Origin_Of,Gilbert,O
    temporal and spatial cropping system,Technology,Used_For,Crop diversification,O
    soil-inhabiting fungus,Organism,Includes,Fusarium oxysporum,Organism
    N,O,Synonym_Of,nitrogen,Nutrient
    hybrid Boro rice,Crop,Seasonal,winter (rabi) season,Season
    agricultural land,O,Includes,tropical forests,Location
    droughts,Natural_Disaster,Caused_By,changing climatic patterns,O
    Oak wilt,Crop,Origin_Of,United States,Location
    genus Botrytis,Organism,Includes,B. cinerea,O
    NARS,Location,Conjunction,AKIS,Organization
    Shade trees,Crop,Includes,ash,O
    habitat intensificatio,O,Origin_Of,Eastern European,Location
    agricultural land,O,Includes,grasslands,Location
    South Eastern region,Location,Includes,Hungary,Location
    agricultural habitats,O,Coreference,habitat intensificatio,O
    RETs,O,Synonym_Of,rice establishment techniques,Agri_Method
    revenue,O,Coreference,Rs. 4180,O
    Mango,Fruit,Includes,Kesar mangoes,Fruit
    Ghana,Location,Includes,Sudan Savannah,Location
    millet crops,Crop,Includes,kodo millet,Crop
    agricultural knowledge and innovation systems,O,Helps_In,Farmer Field School and Landcare,Organization
    irrigation residues,Agri_Waste,Includes,trace metals,O
    P,O,Synonym_Of,Phosphorus,Nutrient
    Agricultural contaminants,Agri_Pollution,Includes,organic matter,Natural_Resource
    South Eastern region,Location,Includes,Romania,Location
    AIS,Location,Synonym_Of,agricultural innovation system,O
    carbon dioxide,Chemical,Synonym_Of,CO2,Chemical
    NDVI,Location,Conjunction,rainfall,O
    insect-borne bacterial infection.,O,Includes,aster yellows,Disease
    healthy soil,Soil,Helps_In,sustaining water quality,O
    fewer Baird’s Sparrow,O,Synonym_Of,Ammodramus bairdii,Organism
    cultivated traditionally,O,Origin_Of,eastern India,Location
    methane,Nutrient,Conjunction,nitrous oxide,Chemical
    soybeans,Crop,Conjunction,maize,Crop
    watermelon,Fruit,Seasonal,Summer,Season
    oaks,Location,Origin_Of,Chinese,O
    barnyard millet,Crop,Includes,sawan,Location
    seven treatments,O,Includes,Soil moisture-based drip irrigation to 60% FC,Agri_Method
    CE,Location,Synonym_Of,Central Europe,Location
    Crown gall,O,Caused_By,Agrobacterium tumefaciens,Organism
    cereal crop,Crop,Includes,Rice,O
    oil palm,Crop,Origin_Of,Latin America,Location
    puddle rice,Crop,Includes,Boro rice,Crop
    Root mean square error,ML_Model,Synonym_Of,RMSE,Organization
    Silvi-Pasture,Crop,Includes,trees,O
    Scirpophaga incertulas,Organism,Synonym_Of,yellow stem borer,Organism
    rice leaf folder,O,Conjunction,yellow stem bore,O
    seven treatments,O,Includes,Conventional drip irrigation at 100% CWR,Agri_Method
    agroforestry models,Agri_Process,Coreference,agroforestry,Agri_Process
    sediments,O,Caused_By,Agricultural,O
    Climate change,Natural_Disaster,Includes,changes in rainfall sequences,O
    staple food crops,Crop,Includes,beans,O
    foliar fungal diseases,Disease,Includes,powdery mildew,Disease
    Birds,O,Origin_Of,Europe,Location
    irrigation residues,Agri_Waste,Includes,salts,O
    agroforestry models,Agri_Process,Includes,AgriHorticulture,Agri_Process
    ASARECA,Organization,Coreference,ASARECA,Organization
    Agricultural contaminants,Agri_Pollution,Includes,insecticides,Chemical
    Spring,Season,Conjunction,Summer,Season
    global climate models,O,Synonym_Of,GCMs,ML_Model
    Root rot,Disease,Caused_By,Phytophthora,Organism
    India,Location,Includes,south-western coastal,Location
    aerobic rice,Crop,Synonym_Of,AR,O
    vegetable transplanting mechanism,Technology,Includes,semiautomatic mechanism,Technology
    rice leaf folder,O,Synonym_Of,Cnaphalocrocis medinalis,Disease
    The United Nations Food and Agriculture Organization,Organization,Synonym_Of,FAO,Organization
    PATs,Other_Quantity,Coreference,PATs,Other_Quantity
    Lesser Grey Shrike,Organism,Coreference,Birds,O
    fungi,Organism,Conjunction,bacteria,Organism
    NARS,Location,Origin_Of,Roseboom,Location
    nutrients,Nutrient,Includes,P,O
    Bacterial wilt,Disease,Caused_By,genera Corynebacterium,Organism
    agricultural habitats,O,Origin_Of,Europe,Location
    Phosphorus,Nutrient,Synonym_Of,P,O
    Agricultural Technology Management Agency,Organization,Synonym_Of,ATMA,Organization
    oil palm,Crop,Origin_Of,Asia,Location
    fungi,Organism,Includes,Clitocybe tabescens,Organism
    agroforestry,Agri_Process,Coreference,agroforestry models,Agri_Process
    West Africa,Location,Origin_Of,tropical forests,Location
    Pennisetum glaucum,Disease,Synonym_Of,pearl millet,Crop
    rose,O,Synonym_Of,Rosaceae,Organism
    Vitamin C,Nutrient,Conjunction,antioxidants,Nutrient
    Agrobacterium tumefaciens,Organism,Synonym_Of,Rhizobium radiobacter,Organism
    Xanthomonas oryzae pathovar oryzae,Organism,Synonym_Of,Xoo,Location
    protection technology improvement,Technology,Conjunction,Input application,O
    dessert,O,Includes,pastries,Food_Item
    agroforestry models,Agri_Process,Includes,Silvi-Pasture,Crop
    Convergence of Sciences,O,Synonym_Of,CoS,Agri_Method
    northwest,O,Synonym_Of,NW,O
    husk,Crop,Conjunction,paddy straw,Agri_Waste
    Rot,O,Caused_By,fungi,Organism
    flood,Natural_Disaster,Caused_By,changing climatic patterns,O
    rice,Crop,Synonym_Of,Oryza sativa L,O
    contaminating the soils,Agri_Pollution,Caused_By,horticultural,Agri_Process
    fodder,O,Conjunction,feed,O
    ZTW,Technology,Synonym_Of,zero tillage wheat,O
    linear mixed models,ML_Model,Used_For,microclimate measurements,Other_Quantity
    AgriHorticulture,Agri_Process,Includes,fruit trees,Crop
    irrigation residues,Agri_Waste,Includes,microorganisms,Organism
    Natural disasters,O,Includes,geophysical disasters,O
    well-fertile aerated soils,Soil,Helps_In,ryegrass,O
    cultivated rice,Crop,Includes,O. glaberrima,O
    Micro-irrigation,Agri_Process,Helps_In,water,Natural_Resource
    N2O,Chemical,Synonym_Of,nitrous oxide,Chemical
    finger millet,Crop,Includes,mandua,Location
    wheat on permanent beds,Crop,Synonym_Of,PBs,Crop
    oxtail millet,Crop,Synonym_Of,Setaria italica,Location
    Mango,Fruit,Includes,Alphonso,Location
    irrigation residues,Agri_Waste,Includes,microorganisms,Organism
    CCS Haryana Agricultural University,Organization,Origin_Of,Haryana,Location
    University of Agricultural Sciences,Organization,Includes,Main Agriculture Research Station,Organization
    spring,Season,Conjunction,summer,Season
    North Eastern region,Location,Includes,Slovak Republic,Location
    crop diversification,Agri_Process,Helps_In,soil moisture,Natural_Resource
    Root rot,Disease,Caused_By,Armillaria mellea,Crop
    nutrients,Nutrient,Includes,Zn,O
    foliar fungal diseases,Disease,Includes,stem canker,Disease
    Mango,Fruit,Includes,Himsagar,Location
    conservation agriculture,O,Synonym_Of,CA,Location
    zero tillage wheat,O,Synonym_Of,ZTW,Technology
    northern part of Brong Ahafo,Location,Conjunction,Uppr East region,Location
    Canker,Location,Caused_By,bacteria,Organism
    hydrometeorological disasters,Natural_Disaster,Includes,droughts,Natural_Disaster
    Eastern European,Location,Conjunction,Mediterranean countries,Location
    Crown gall,O,Caused_By,Rhizobium radiobacter,Organism
    ICAR- Central Soil Salinity Research Institute,Organization,Synonym_Of,CSSRI,Organization
    Guava,Location,Seasonal,summer,Season
    sediments,O,Caused_By,horticultural,Agri_Process
    2018–19,Duration,Conjunction,2017–18,Duration
    Canker,Location,Caused_By,fungi,Organism
    CoS,Agri_Method,Synonym_Of,Convergence of Sciences,O
    apricots,Fruit,Seasonal,spring,Season
    1000 mm,Rainfall,Coreference,500 – 700 mm,Rainfall
    Karnataka,Location,Includes,Dharwad,Location
    agronomic crops,Crop,Includes,medicinal crops,Crop
    fungi,Organism,Includes,Armillaria mellea,Crop
    oxtail millet,Crop,Includes,kangni,Location
    bacterial blight,Disease,Origin_Of,western coast of Africa,Location
    NARS,Location,Conjunction,AKIS,Organization
    Climate change,Natural_Disaster,Includes,movement of climatic regions,O
    Maximum entropy,O,Synonym_Of,MaxEnt,ML_Model
    methods of crop farming,O,Conjunction,adapted cropping pattern,O
    bacteria,Organism,Conjunction,fungi,Organism
    millets,Crop,Includes,Teff,O
    Agri-Horti-Silviculture,Agri_Process,Includes,fruit trees,Crop
    plant decomposition,O,Conjunction,putrefaction,Agri_Process
    AIS,Location,Synonym_Of,Agricultural Innovation Systems,Organization
    crop diversification,Agri_Process,Helps_In,small landholder farmers,Person
    contaminating the soils,Agri_Pollution,Caused_By,horticultural,Agri_Process
    Lesser Grey Shrike,Organism,Origin_Of,Western Europe,Location
    storms,O,Caused_By,changing climatic patterns,O
    dessert,O,Includes,cherry sauces,Food_Item
    silage,Agri_Waste,Conjunction,hay,O
    changes in rainfall sequences,O,Conjunction,movement of climatic regions,O
    Agricultural,O,Conjunction,horticultural,Agri_Process
    Fusarium wilt,Disease,Caused_By,soil-inhabiting fungus,Organism
    oomycetes,Chemical,Includes,Aphanomyces,O
    Agri-Horti-Silviculture,Agri_Process,Includes,trees,O
    Horti-Pasture,Food_Item,Includes,fodder crops,Crop
    Gray mold rot,Disease,Caused_By,fungi,Organism
    foliar fungal diseases,Disease,Includes,leaf rot,Disease
    Europe,Location,Coreference,Europe,Location
    microclimate,Agri_Process,Conjunction,soil moisture fluctuation,O
    oaks,Location,Includes,Quercus,Disease
    multi-stakeholders’ approach,O,Helps_In,farmers,Person
    grapes,O,Coreference,green grapes,Fruit
    organic matter,Natural_Resource,Includes,decaying plant material,O
    Latin America,Location,Conjunction,West Africa,Location
    CH4,O,Synonym_Of,methane,Nutrient
    IGP,Organization,Synonym_Of,Indo-Gangetic Plains of India,Field_Area
    Microclimate,Agri_Process,Conjunction,soil moisture,Natural_Resource
    sediments,O,Caused_By,horticultural,Agri_Process
    Micro-irrigation,Agri_Process,Helps_In,money,O
    Ferterra,Location,Includes,chlorantraniliprole,Chemical
    Sorghum bicolor,Crop,Synonym_Of,sorghum,Crop
    ASARECA,Organization,Conjunction,CORAF,Organization
    Farmers,Person,Helps_In,arm management,O
    herbicides,Chemical,Conjunction,insecticides,Chemical
    Input application,O,Conjunction,protection technology improvement,Technology
    Regional Research Station,Organization,Origin_Of,Bawal,Location
    CO2,Chemical,Caused_By,Human,O
    apple,Fruit,Origin_Of,Kashmir region,Location
    Boro rice,Crop,Seasonal,winter,Season
    insect-proof net buildings,Technology,Helps_In,virus-free crop production,Agri_Process
    pomegranate,Fruit,Seasonal,winter,Season
    sorghum,Crop,Includes,jowar,Location
    green apple,Fruit,Coreference,apple,Fruit
    N,O,Synonym_Of,nitrogen,Nutrient
    brown top millet,Crop,Includes,korale,Crop
    CORAF,Organization,Conjunction,SADDC,Organization
    Oryza sativa L,O,Synonym_Of,rice,Crop
    nutrients,Nutrient,Includes,Mn,O
    nutrients,Nutrient,Includes,Si,O
    Romania,Location,Conjunction,Bulgaria,Location
    2017–18,Duration,Conjunction,2018–19,Duration
    barnyard millet,Crop,Includes,jhangora,Location
    oil palm,Crop,Origin_Of,tropical forest,Location
    irrigation methods,O,Conjunction,methods of crop farming,O
    LMT,Technology,Synonym_Of,Logistic Model Trees,ML_Model
    scab,O,Caused_By,bacteria,Organism
    agroforestry models,Agri_Process,Includes,Horti-Pasture,Food_Item
    apple,Fruit,Coreference,Apples,Fruit
    wilt diseases,Disease,Caused_By,viruses,O
    microclimate,Agri_Process,Coreference,Microclimate,Agri_Process
    seven treatments,O,Includes,Timer based drip irrigation to 60% CWR,Agri_Method
    India,Location,Includes,Karnal,Location
    2019–20,Duration,Conjunction,2020–21,Duration
    organic matter,Natural_Resource,Includes,decaying plant material,O
    Apples,Fruit,Coreference,apple,Fruit
    plant diseases,O,Caused_By,classical pesticides,Chemical
    bacterial blight,Disease,Origin_Of,Latin America,Location
    ASARECA,Organization,Coreference,ASARECA,Organization
    CSSRI,Organization,Synonym_Of,ICAR- Central Soil Salinity Research Institute,Organization
    agroforestry models,Agri_Process,Includes,Homestead Agroforestry,Organization
    organic matter,Natural_Resource,Includes,animal wastes,Agri_Waste
    malnutrition,O,Conjunction,under-nutrition,O
    Gray mold rot,Disease,Caused_By,B. cinerea,O
    500 – 700 mm,Rainfall,Coreference,1000 mm,Rainfall
    water,Natural_Resource,Used_For,Hydroponic fodder,Crop
    healthy soil,Soil,Helps_In,removing greenhouse gases from the atmosphere,O
    apple,Fruit,Origin_Of,Europe,Location
    tropical ecosystems,Location,Coreference,tropical forests,Location
    Heavy metals,Crop,Includes,Cu,Location
    seven treatments,O,Includes,Soil moisture-based drip irrigation to 80% FC,Agri_Method
    CO2,Chemical,Synonym_Of,carbon dioxide,Chemical
    Jackfruit,Fruit,Seasonal,summer,Season
    Agricultural contaminants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    microclimate,Agri_Process,Conjunction,soil moisture,Natural_Resource
    agricultural knowledge and innovation systems,O,Synonym_Of,AKIS,Organization
    R2,O,Includes,coefficient of determination,O
    Ammodramus bairdii,Organism,Synonym_Of,fewer Baird’s Sparrow,O
    Conservation tillage,O,Helps_In,grower’s profitability,O
    agricultural intensifiation,Agri_Process,Origin_Of,Western Europe,Location
    proso millet,O,Synonym_Of,Setaria italica,Location
    hydrometeorological disasters,Natural_Disaster,Includes,cyclones,Natural_Disaster
    Micro-irrigation,Agri_Process,Helps_In,farmers’ income,Person
    habitat intensificatio,O,Origin_Of,Mediterranean countries,Location
    hydrometeorological disasters,Natural_Disaster,Includes,heatwaves,Natural_Disaster
    staple food crops,Crop,Includes,rice,Crop
    agroforestry models,Agri_Process,Includes,Agri-Horti-Silviculture,Agri_Process
    microorganisms,Organism,Helps_In,soil health,O
    food crops,Crop,Includes,tomatoes,Crop
    NW,O,Synonym_Of,northwest,O
    South Eastern region,Location,Includes,Macedonia,Location
    kodo millet,Crop,Synonym_Of,Paspalum scrobiculatum,Crop
    antioxidants,Nutrient,Conjunction,Vitamin C,Nutrient
    apple,Fruit,Origin_Of,New Zealand,Location
    Agricultural contaminants,Agri_Pollution,Includes,irrigation residues,Agri_Waste
    Fire blight,Disease,Caused_By,warm moist weather,O
    increased temperatures,O,Caused_By,movement of climatic regions,O
    Calcarius ornatus,Organism,Synonym_Of,Chestnut-collared Longspur,Crop
    NEH,Location,Synonym_Of,North-Eastern Hills,Location
    summer,Season,Conjunction,spring,Season
    North-Eastern Hills,Location,Synonym_Of,NEH,Location
    Jackfruit,Fruit,Seasonal,spring,Season
    Agri-Silviculture,Agri_Process,Includes,trees,O
    FAO,Organization,Synonym_Of,The United Nations Food and Agriculture Organization,Organization
    pearl millet,Crop,Synonym_Of,Pennisetum glaucum,Disease
    oaks,Location,Origin_Of,American,O
    resources,O,Coreference,water,Natural_Resource
    soil moisture,Natural_Resource,Coreference,soil moisture fluctuation,O
    Eleusine coracana,Crop,Synonym_Of,finger millet,Crop
    CO2,Chemical,Coreference,CO2,Chemical
    nutrients,Nutrient,Includes,Cl,O
    biomass harvesting,Agri_Process,Coreference,biomass harvesting,Agri_Process
    processes,O,Coreference,habitat intensificatio,O
    ATMA,Organization,Synonym_Of,Agricultural Technology Management Agency,Organization
    National Research Council,Organization,Origin_Of,U.S.,Location
    2018,Date_and_Time,Conjunction,2019,Date_and_Time
    stovers,Other,Conjunction,residues-straw,Agri_Waste
    whole farm planning toolkits,O,Conjunction,integrated pest management,O
    residues-straw,Agri_Waste,Conjunction,stovers,Other
    rice establishment techniques,Agri_Method,Synonym_Of,RETs,O
    acervuli,O,Caused_By,fungi,Organism
    Horti-Pasture,Food_Item,Includes,fruit trees,Crop
    multidisciplinary approach,O,Conjunction,robust economic model,O
    Setaria italica,Location,Synonym_Of,oxtail millet,Crop
    biomass harvesting,Agri_Process,Coreference,biomass harvesting,Agri_Process
    agronomic crops,Crop,Includes,sugar crops,Crop
    Uppr East region,Location,Conjunction,northern part of Brong Ahafo,Location
    horticultural,Agri_Process,Conjunction,Agricultural,O
    Polluting the soil,O,Caused_By,horticultural practices,Agri_Process
    Soil contamination,Agri_Pollution,Caused_By,horticultural activities,Agri_Process
    Causing soil pollution,O,Caused_By,horticulture,Agri_Process
    Soil pollution,Agri_Pollution,Caused_By,horticultural methods,Agri_Process
    Contaminating the land,O,Caused_By,horticultural practices,Agri_Process
    contaminating the soils,Agri_Pollution,Caused_By,horticultural,Agri_Process
    debris,O,Caused_By,horticultural activities,Agri_Process
    silt,O,Caused_By,horticultural practices,Agri_Process
    soil erosion,O,Caused_By,horticultural,Agri_Process
    sedimentation,O,Caused_By,horticultural,Agri_Process
    alluvium,Crop,Caused_By,horticultural activities,Agri_Process
    sediments,O,Caused_By,horticultural,Agri_Process
    polluting the land,O,Caused_By,Agricultural,O
    soil contamination,O,Caused_By,Agricultural,O
    contaminating the earth,O,Caused_By,Agricultural,O
    land pollution,O,Caused_By,Agricultural,O
    soil pollution,O,Caused_By,Agricultural,O
    contaminating the soils,Agri_Pollution,Caused_By,Agricultural,O
    sediment deposits,O,Caused_By,Agricultural practices,O
    sedimentary materials,O,Caused_By,Agricultural activities,O
    deposited sediments,O,Caused_By,Agriculture,Agri_Process
    sedimentation,O,Caused_By,Farming practices,O
    sediment buildup,O,Caused_By,Agricultural operations,O
    sediments,O,Caused_By,Agricultural,O
    Agri_Pollution,Agri_Process,Includes,natural and synthetic fertilizers,O
    Agricultural pollutants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Contaminants,Chemical,Includes,natural and synthetic fertilizers,O
    Agricultural pollution,O,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollution,Agri_Process,Includes,bio and chemical fertilizers,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Farming pollutants,Agri_Pollution,Includes,pesticides,Chemical
    Agri_Pollution,Agri_Process,Includes,pesticides,Chemical
    Agricultural pollutants,Agri_Pollution,Includes,pesticides,Chemical
    Farm contaminants,Agri_Pollution,Includes,pesticides,Chemical
    Agricultural pollution,O,Includes,pesticides,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,pesticides,Chemical
    Agri_Pollution,Agri_Process,Includes,weed killers,O
    Agricultural pollutants,Agri_Pollution,Includes,herbicides,Chemical
    Farm pollution,O,Includes,herbicides,Chemical
    Agri_Pollution,Agri_Process,Includes,weed control chemicals,Chemical
    Agricultural contamination,Disease,Includes,herbicides,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,herbicides,Chemical
    gardening,O,Conjunction,farming,O
    horticultural,Agri_Process,Coreference,gardening,O
    Agricultural,O,Coreference,farming,O
    horticultural,Agri_Process,Conjunction,agriculture,Agri_Process
    horticultural,Agri_Process,Coreference,horticultural,Agri_Process
    Agricultural,O,Coreference,agriculture,Agri_Process
    plant cultivation,Agri_Process,Conjunction,agricultural,O
    horticultural,Agri_Process,Coreference,plant cultivation,Agri_Process
    Agricultural,O,Coreference,agricultural,O
    Horticultural,O,Conjunction,Agricultural,O
    horticultural,Agri_Process,Coreference,Horticultural,O
    Plant cultivation,Agri_Process,Conjunction,Agricultural,O
    horticultural,Agri_Process,Coreference,Plant cultivation,Agri_Process
    silt,O,Caused_By,horticultural,Agri_Process
    sediments,O,Coreference,silt,O
    alluvium,Crop,Caused_By,horticultural,Agri_Process
    sediments,O,Coreference,alluvium,Crop
    soil contamination,O,Caused_By,Agricultural,O
    contaminating the soils,Agri_Pollution,Coreference,soil contamination,O
    polluting the land,O,Caused_By,Agricultural,O
    contaminating the soils,Agri_Pollution,Coreference,polluting the land,O
    soil pollution,O,Caused_By,Agricultural,O
    contaminating the soils,Agri_Pollution,Coreference,soil pollution,O
    contaminating the earth,O,Caused_By,Agricultural,O
    contaminating the soils,Agri_Pollution,Coreference,contaminating the earth,O
    land pollution,O,Caused_By,Agricultural,O
    contaminating the soils,Agri_Pollution,Coreference,land pollution,O
    Agricultural pollutants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agricultural contaminants,Agri_Pollution,Coreference,Agricultural pollutants,Agri_Pollution
    Farming contaminants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agricultural contaminants,Agri_Pollution,Coreference,Farming contaminants,Agri_Pollution
    Gardening,O,Conjunction,Farming,O
    horticultural,Agri_Process,Coreference,Gardening,O
    Agricultural,O,Coreference,Farming,O
    Plant cultivation,Agri_Process,Conjunction,Agriculture,Agri_Process
    horticultural,Agri_Process,Coreference,Plant cultivation,Agri_Process
    Agricultural,O,Coreference,Agriculture,Agri_Process
    Crop cultivation,Agri_Process,Conjunction,Agricultural,O
    horticultural,Agri_Process,Coreference,Crop cultivation,Agri_Process
    Agricultural,O,Coreference,Agricultural,O
    Horticulture,Agri_Process,Conjunction,Farming,O
    horticultural,Agri_Process,Coreference,Horticulture,Agri_Process
    Agricultural,O,Coreference,Farming,O
    Horticultural practice,O,Conjunction,Agriculture,Agri_Process
    horticultural,Agri_Process,Coreference,Horticultural practice,O
    Agricultural,O,Coreference,Agriculture,Agri_Process
    Polluting the soil,O,Caused_By,horticulture,Agri_Process
    contaminating the soils,Agri_Pollution,Coreference,Polluting the soil,O
    horticultural,Agri_Process,Coreference,horticulture,Agri_Process
    Soil contamination,Agri_Pollution,Caused_By,horticultural practices,Agri_Process
    contaminating the soils,Agri_Pollution,Coreference,Soil contamination,Agri_Pollution
    horticultural,Agri_Process,Coreference,horticultural practices,Agri_Process
    Soil pollution,Agri_Pollution,Caused_By,horticulture,Agri_Process
    contaminating the soils,Agri_Pollution,Coreference,Soil pollution,Agri_Pollution
    horticultural,Agri_Process,Coreference,horticulture,Agri_Process
    Contaminating the land,O,Caused_By,horticultural activities,Agri_Process
    contaminating the soils,Agri_Pollution,Coreference,Contaminating the land,O
    horticultural,Agri_Process,Coreference,horticultural activities,Agri_Process
    Soil degradation,Agri_Pollution,Caused_By,horticulture,Agri_Process
    contaminating the soils,Agri_Pollution,Coreference,Soil degradation,Agri_Pollution
    horticultural,Agri_Process,Coreference,horticulture,Agri_Process
    soil erosion,O,Caused_By,horticultural practices,Agri_Process
    sediments,O,Coreference,soil erosion,O
    horticultural,Agri_Process,Coreference,horticultural practices,Agri_Process
    sediments,O,Caused_By,gardening activities,O
    sediments,O,Coreference,sediments,O
    horticultural,Agri_Process,Coreference,gardening activities,O
    deposits,O,Caused_By,horticultural,Agri_Process
    sediments,O,Coreference,deposits,O
    horticultural,Agri_Process,Coreference,horticultural,Agri_Process
    sediments,O,Caused_By,cultivation,Agri_Process
    sediments,O,Coreference,sediments,O
    horticultural,Agri_Process,Coreference,cultivation,Agri_Process
    polluting the land,O,Caused_By,Farming,O
    contaminating the soils,Agri_Pollution,Coreference,polluting the land,O
    Agricultural,O,Coreference,Farming,O
    contaminating the earth,O,Caused_By,Agricultural activities,O
    contaminating the soils,Agri_Pollution,Coreference,contaminating the earth,O
    Agricultural,O,Coreference,Agricultural activities,O
    soil contamination,O,Caused_By,Agriculture,Agri_Process
    contaminating the soils,Agri_Pollution,Coreference,soil contamination,O
    Agricultural,O,Coreference,Agriculture,Agri_Process
    causing soil pollution,O,Caused_By,Agro-based,Agri_Method
    contaminating the soils,Agri_Pollution,Coreference,causing soil pollution,O
    Agricultural,O,Coreference,Agro-based,Agri_Method
    dirtying the grounds,O,Caused_By,Farming practices,O
    contaminating the soils,Agri_Pollution,Coreference,dirtying the grounds,O
    Agricultural,O,Coreference,Farming practices,O
    sedimentation,O,Caused_By,Agricultural activities,O
    sediments,O,Coreference,sedimentation,O
    Agricultural,O,Coreference,Agricultural activities,O
    deposits,O,Caused_By,Agricultural practices,O
    sediments,O,Coreference,deposits,O
    Agricultural,O,Coreference,Agricultural practices,O
    soil particles,O,Caused_By,Agricultural impact,O
    sediments,O,Coreference,soil particles,O
    Agricultural,O,Coreference,Agricultural impact,O
    residues,O,Caused_By,Agricultural influence,O
    sediments,O,Coreference,residues,O
    Agricultural,O,Coreference,Agricultural influence,O
    settled materials,O,Caused_By,Agricultural effects,O
    sediments,O,Coreference,settled materials,O
    Agricultural,O,Coreference,Agricultural effects,O
    Agri_Pollution,Agri_Process,Includes,natural and synthetic fertilizers,O
    Agricultural contaminants,Agri_Pollution,Coreference,Agri_Pollution,Agri_Process
    organic and inorganic fertilizers,Chemical,Coreference,natural and synthetic fertilizers,O
    Agricultural pollutants,Agri_Pollution,Includes,natural and synthetic fertilizers,O
    Agricultural contaminants,Agri_Pollution,Coreference,Agricultural pollutants,Agri_Pollution
    organic and inorganic fertilizers,Chemical,Coreference,natural and synthetic fertilizers,O
    Agri_Pollution,Agri_Process,Includes,natural and synthetic fertilizers,O
    Agricultural pollutants,Agri_Pollution,Includes,natural and synthetic fertilizers,O
    Agri_Pollution,Agri_Process,Includes,natural and synthetic fertilizers,O
    Agri_Pollution,Agri_Process,Includes,weed killers,O
    Agricultural contaminants,Agri_Pollution,Coreference,Agri_Pollution,Agri_Process
    herbicides,Chemical,Coreference,weed killers,O
    Agricultural pollutants,Agri_Pollution,Includes,herbicides,Chemical
    Agricultural contaminants,Agri_Pollution,Coreference,Agricultural pollutants,Agri_Pollution
    herbicides,Chemical,Coreference,herbicides,Chemical
    Farm pollution,O,Includes,weed killers,O
    Agricultural contaminants,Agri_Pollution,Coreference,Farm pollution,O
    herbicides,Chemical,Coreference,weed killers,O
    Agri_Pollution,Agri_Process,Includes,herbicidal substances,O
    Agricultural contaminants,Agri_Pollution,Coreference,Agri_Pollution,Agri_Process
    herbicides,Chemical,Coreference,herbicidal substances,O
    Agricultural contamination,Disease,Includes,herbicides,Chemical
    Agricultural contaminants,Agri_Pollution,Coreference,Agricultural contamination,Disease
    herbicides,Chemical,Coreference,herbicides,Chemical
    Polluting the land,O,Caused_By,horticulture,Agri_Process
    contaminating the soils,Agri_Pollution,Coreference,Polluting the land,O
    horticultural,Agri_Process,Coreference,horticulture,Agri_Process
    Soil pollution,Agri_Pollution,Caused_By,horticultural,Agri_Process
    contaminating the soils,Agri_Pollution,Coreference,Soil pollution,Agri_Pollution
    Contaminating the earth,O,Caused_By,horticulture,Agri_Process
    contaminating the soils,Agri_Pollution,Coreference,Contaminating the earth,O
    horticultural,Agri_Process,Coreference,horticulture,Agri_Process
    Causing soil contamination,O,Caused_By,horticultural,Agri_Process
    contaminating the soils,Agri_Pollution,Coreference,Causing soil contamination,O
    Land pollution,O,Caused_By,horticulture,Agri_Process
    contaminating the soils,Agri_Pollution,Coreference,Land pollution,O
    horticultural,Agri_Process,Coreference,horticulture,Agri_Process
    debris,O,Caused_By,horticultural,Agri_Process
    sediments,O,Coreference,debris,O
    Agri_Pollution,Agri_Process,Includes,pest control substances,Agri_Pollution
    Agricultural contaminants,Agri_Pollution,Coreference,Agri_Pollution,Agri_Process
    pesticides,Chemical,Coreference,pest control substances,Agri_Pollution
    Agri_Pollution,Agri_Process,Includes,weed killers,O
    Agricultural contaminants,Agri_Pollution,Coreference,Agri_Pollution,Agri_Process
    herbicides,Chemical,Coreference,weed killers,O
    Agricultural pollutants,Agri_Pollution,Includes,herbicides,Chemical
    Agricultural contaminants,Agri_Pollution,Coreference,Agricultural pollutants,Agri_Pollution
    Farming pollutants,Agri_Pollution,Includes,weed killers,O
    Agricultural contaminants,Agri_Pollution,Coreference,Farming pollutants,Agri_Pollution
    herbicides,Chemical,Coreference,weed killers,O
    Agri_Pollution,Agri_Process,Includes,herbicidal substances,O
    Agricultural contaminants,Agri_Pollution,Coreference,Agri_Pollution,Agri_Process
    herbicides,Chemical,Coreference,herbicidal substances,O
    Agricultural contaminants,Agri_Pollution,Includes,weed killers,O
    herbicides,Chemical,Coreference,weed killers,O
    Agri_Pollution,Agri_Process,Includes,pesticides,Chemical
    Agricultural contaminants,Agri_Pollution,Coreference,Agri_Pollution,Agri_Process
    insecticides,Chemical,Coreference,pesticides,Chemical
    Agricultural pollutants,Agri_Pollution,Includes,insecticidal substances,O
    Agricultural contaminants,Agri_Pollution,Coreference,Agricultural pollutants,Agri_Pollution
    insecticides,Chemical,Coreference,insecticidal substances,O
    Agri_Waste,Agri_Process,Includes,insecticides,Chemical
    Agricultural contaminants,Agri_Pollution,Coreference,Agri_Waste,Agri_Process
    Farm contaminants,Agri_Pollution,Includes,insecticides,Chemical
    Agricultural contaminants,Agri_Pollution,Coreference,Farm contaminants,Agri_Pollution
    Agriculture pollution,O,Includes,insecticides,Chemical
    Agricultural contaminants,Agri_Pollution,Coreference,Agriculture pollution,O
    irrigation leftovers,O,Includes,salts,O
    irrigation residues,Agri_Waste,Coreference,irrigation leftovers,O
    irrigation waste,O,Includes,salts,O
    irrigation residues,Agri_Waste,Coreference,irrigation waste,O
    irrigation remnants,O,Includes,salts,O
    irrigation residues,Agri_Waste,Coreference,irrigation remnants,O
    irrigation byproducts,Agri_Waste,Includes,salts,O
    irrigation residues,Agri_Waste,Coreference,irrigation byproducts,Agri_Waste
    irrigation debris,O,Includes,salts,O
    irrigation residues,Agri_Waste,Coreference,irrigation debris,O
    irrigation waste,O,Includes,microbes,Organism
    irrigation residues,Agri_Waste,Coreference,irrigation waste,O
    microorganisms,Organism,Coreference,microbes,Organism
    irrigation byproducts,Agri_Waste,Includes,bugs,O
    irrigation residues,Agri_Waste,Coreference,irrigation byproducts,Agri_Waste
    microorganisms,Organism,Coreference,bugs,O
    agricultural runoff,O,Includes,bacteria,Organism
    irrigation residues,Agri_Waste,Coreference,agricultural runoff,O
    microorganisms,Organism,Coreference,bacteria,Organism
    farming residues,O,Includes,germs,Disease
    irrigation residues,Agri_Waste,Coreference,farming residues,O
    microorganisms,Organism,Coreference,germs,Disease
    irrigation leftovers,O,Includes,microscopic organisms,Organism
    irrigation residues,Agri_Waste,Coreference,irrigation leftovers,O
    microorganisms,Organism,Coreference,microscopic organisms,Organism
    Nitrogen,Nutrient,Helps_In,crop production,O
    nitrogen,Nutrient,Coreference,Nitrogen,Nutrient
    crop yield,O,Coreference,crop production,O
    Nitrogen element,Nutrient,Helps_In,crop yield,O
    nitrogen,Nutrient,Coreference,Nitrogen element,Nutrient
    N,O,Helps_In,crop productivity,O
    nitrogen,Nutrient,Coreference,N,O
    crop yield,O,Coreference,crop productivity,O
    Nitrogenous,O,Helps_In,crop yield,O
    nitrogen,Nutrient,Coreference,Nitrogenous,O
    Nitrogen,Nutrient,Helps_In,crop output,O
    Polluting the farmlands,O,Caused_By,horticultural,Agri_Process
    contaminating the soils,Agri_Pollution,Coreference,Polluting the farmlands,O
    Causing soil contamination,O,Caused_By,horticultural,Agri_Process
    contaminating the soils,Agri_Pollution,Coreference,Causing soil contamination,O
    Contaminating agricultural lands,O,Caused_By,horticultural,Agri_Process
    contaminating the soils,Agri_Pollution,Coreference,Contaminating agricultural lands,O
    Soil pollution resulted from,Agri_Pollution,Caused_By,horticultural,Agri_Process
    contaminating the soils,Agri_Pollution,Coreference,Soil pollution resulted from,Agri_Pollution
    Contaminating the croplands,O,Caused_By,horticultural,Agri_Process
    contaminating the soils,Agri_Pollution,Coreference,Contaminating the croplands,O
    sedimentary deposits,O,Caused_By,horticultural practices,Agri_Process
    sediments,O,Coreference,sedimentary deposits,O
    horticultural,Agri_Process,Coreference,horticultural practices,Agri_Process
    deposits,O,Caused_By,horticulture,Agri_Process
    sediments,O,Coreference,deposits,O
    horticultural,Agri_Process,Coreference,horticulture,Agri_Process
    sediment,O,Caused_By,horticultural activities,Agri_Process
    sediments,O,Coreference,sediment,O
    horticultural,Agri_Process,Coreference,horticultural activities,Agri_Process
    settling particles,O,Caused_By,horticultural,Agri_Process
    sediments,O,Coreference,settling particles,O
    Polluting the farmland,O,Caused_By,industrial activities,O
    contaminating the soils,Agri_Pollution,Coreference,Polluting the farmland,O
    Soil contamination,Agri_Pollution,Caused_By,industrial practices,O
    contaminating the soils,Agri_Pollution,Coreference,Soil contamination,Agri_Pollution
    industrial activities,O,Coreference,industrial practices,O
    Contaminating agricultural lands,O,Caused_By,industrial activities,O
    contaminating the soils,Agri_Pollution,Coreference,Contaminating agricultural lands,O
    Soil pollution,Agri_Pollution,Caused_By,industrial activities,O
    contaminating the soils,Agri_Pollution,Coreference,Soil pollution,Agri_Pollution
    Soil tainting,Soil,Caused_By,industrial activities,O
    contaminating the soils,Agri_Pollution,Coreference,Soil tainting,Soil
    soil erosion,O,Caused_By,manufacturing processes,O
    sediments,O,Coreference,soil erosion,O
    industrial activities,O,Coreference,manufacturing processes,O
    debris,O,Caused_By,factory operations,O
    sediments,O,Coreference,debris,O
    industrial activities,O,Coreference,factory operations,O
    silt,O,Caused_By,industrial practices,O
    sediments,O,Coreference,silt,O
    industrial activities,O,Coreference,industrial practices,O
    particulate matter,Agri_Pollution,Caused_By,commercial activities,O
    sediments,O,Coreference,particulate matter,Agri_Pollution
    industrial activities,O,Coreference,commercial activities,O
    alluvium,Crop,Caused_By,industrial developments,O
    sediments,O,Coreference,alluvium,Crop
    industrial activities,O,Coreference,industrial developments,O
    Nutrient,Nutrient,Synonym_Of,Phosphorus,Nutrient
    P,O,Coreference,Nutrient,Nutrient
    Essential element,O,Synonym_Of,Phosphorus,Nutrient
    P,O,Coreference,Essential element,O
    Mineral,Natural_Resource,Synonym_Of,Phosphorus,Nutrient
    P,O,Coreference,Mineral,Natural_Resource
    P,O,Synonym_Of,Phosphorus,Nutrient
    Phosphorus,Nutrient,Coreference,Phosphorus,Nutrient
    Plant nutrient,Nutrient,Synonym_Of,Phosphorus,Nutrient
    P,O,Coreference,Plant nutrient,Nutrient
    Nutrient,Nutrient,Used_For,crop cultivation,Agri_Process
    P,O,Coreference,Nutrient,Nutrient
    crop production,O,Coreference,crop cultivation,Agri_Process
    Nutrient source,O,Used_For,crop production,O
    P,O,Coreference,Nutrient source,O
    Essential nutrient,O,Used_For,crop growth,O
    P,O,Coreference,Essential nutrient,O
    crop production,O,Coreference,crop growth,O
    Nutrient supply,Nutrient,Used_For,crop farming,Agri_Process
    P,O,Coreference,Nutrient supply,Nutrient
    crop production,O,Coreference,crop farming,Agri_Process
    Nutrient element,Nutrient,Used_For,crop production,O
    P,O,Coreference,Nutrient element,Nutrient
    Phosphorus,Nutrient,Synonym_Of,phosphate,O
    P,O,Coreference,Phosphorus,Nutrient
    phosphorus,Natural_Resource,Coreference,phosphate,O
    Nutrient,Nutrient,Synonym_Of,phosphorus,Natural_Resource
    P,O,Coreference,Nutrient,Nutrient
    P,O,Synonym_Of,phosphate compound,Nutrient
    phosphorus,Natural_Resource,Coreference,phosphate compound,Nutrient
    Nutrient,Nutrient,Synonym_Of,phosphate element,Nutrient
    P,O,Coreference,Nutrient,Nutrient
    phosphorus,Natural_Resource,Coreference,phosphate element,Nutrient
    FAO,Organization,Synonym_Of,Food and Agriculture Institution,Organization
    Food and Agriculture Organization,Organization,Coreference,Food and Agriculture Institution,Organization
    Food and Agriculture Organization,Organization,Synonym_Of,FAO,Organization
    FAO,Organization,Coreference,Food and Agriculture Organization,Organization
    Food and Agriculture Organization,Organization,Coreference,FAO,Organization
    FAO,Organization,Synonym_Of,Food and Farming Organization,Organization
    Food and Agriculture Organization,Organization,Coreference,Food and Farming Organization,Organization
    Food and Agriculture Organization,Organization,Synonym_Of,FAO,Organization
    FAO,Organization,Coreference,Food and Agriculture Organization,Organization
    Food and Agriculture Organization,Organization,Coreference,FAO,Organization
    FAO,Organization,Synonym_Of,Food and Agriculture Agency,Organization
    Food and Agriculture Organization,Organization,Coreference,Food and Agriculture Agency,Organization
    FAO,Organization,Coreference,Food and Agriculture Organization,Organization
    FAO,Organization,Coreference,FAO,Organization
    Food and Agriculture Organization,Organization,Coreference,FAO,Organization
    Food and Agriculture Organization,Organization,Synonym_Of,FAO,Organization
    FAO,Organization,Coreference,Food and Agriculture Organization,Organization
    Food and Agriculture Organization,Organization,Coreference,FAO,Organization
    oilseed palm,Crop,Origin_Of,Asia,Location
    oil palm,Crop,Coreference,oilseed palm,Crop
    palm oil,Crop,Origin_Of,Asia,Location
    oil palm,Crop,Coreference,palm oil,Crop
    oil-bearing palm,Crop,Origin_Of,Asia,Location
    oil palm,Crop,Coreference,oil-bearing palm,Crop
    palm oleaginous plant,Crop,Origin_Of,Asia,Location
    oil palm,Crop,Coreference,palm oleaginous plant,Crop
    oleaginous palm,Disease,Origin_Of,Asia,Location
    oil palm,Crop,Coreference,oleaginous palm,Disease
    oil palm tree,Crop,Origin_Of,West Africa,Location
    oil palm,Crop,Coreference,oil palm tree,Crop
    palm oil,Crop,Origin_Of,West Africa,Location
    oil palm,Crop,Coreference,palm oil,Crop
    African oil palm,Crop,Origin_Of,West Africa,Location
    oil palm,Crop,Coreference,African oil palm,Crop
    oil-producing palm,Crop,Origin_Of,West Africa,Location
    oil palm,Crop,Coreference,oil-producing palm,Crop
    palm nut tree,Crop,Origin_Of,West Africa,Location
    oil palm,Crop,Coreference,palm nut tree,Crop
    Farmed area,O,Includes,tropical forests,Location
    agricultural land,O,Coreference,Farmed area,O
    Cultivated land,O,Includes,tropical forests,Location
    agricultural land,O,Coreference,Cultivated land,O
    Agricultural area,O,Includes,tropical forests,Location
    agricultural land,O,Coreference,Agricultural area,O
    Farmed area,O,Includes,grasslands,Location
    agricultural land,O,Coreference,Farmed area,O
    Agricultural region,Location,Includes,grasslands,Location
    agricultural land,O,Coreference,Agricultural region,Location
    Farmland,Location,Includes,grasslands,Location
    agricultural land,O,Coreference,Farmland,Location
    Agricultural territory,Location,Includes,grasslands,Location
    agricultural land,O,Coreference,Agricultural territory,Location
    Agricultural land,O,Includes,meadows,Field_Area
    agricultural land,O,Coreference,Agricultural land,O
    grasslands,Location,Coreference,meadows,Field_Area
    Farming area,Location,Includes,savannas,Fruit
    agricultural land,O,Coreference,Farming area,Location
    Farmland,Location,Includes,savannas,Fruit
    agricultural land,O,Coreference,Farmland,Location
    Agricultural region,Location,Includes,savannas,Fruit
    agricultural land,O,Coreference,Agricultural region,Location
    Cultivated area,O,Includes,savannas,Fruit
    agricultural land,O,Coreference,Cultivated area,O
    Agricultural territory,Location,Includes,savannas,Fruit
    agricultural land,O,Coreference,Agricultural territory,Location
    Farmed area,O,Includes,shrublands,Location
    agricultural land,O,Coreference,Farmed area,O
    Agricultural region,Location,Includes,shrublands,Location
    agricultural land,O,Coreference,Agricultural region,Location
    Farmland,Location,Includes,shrublands,Location
    agricultural land,O,Coreference,Farmland,Location
    Cultivated land,O,Includes,shrublands,Location
    agricultural land,O,Coreference,Cultivated land,O
    Agricultural property,O,Includes,shrublands,Location
    agricultural land,O,Coreference,Agricultural property,O
    Avian creatures,Organism,Origin_Of,Europe,Location
    Birds,O,Coreference,Avian creatures,Organism
    Feathered animals,Organism,Origin_Of,Europe,Location
    Birds,O,Coreference,Feathered animals,Organism
    Flying creatures,Organism,Origin_Of,Europe,Location
    Birds,O,Coreference,Flying creatures,Organism
    Winged beings,Organism,Origin_Of,Europe,Location
    Birds,O,Coreference,Winged beings,Organism
    weed killers,O,Conjunction,insect repellents,Chemical
    herbicides,Chemical,Coreference,weed killers,O
    insecticides,Chemical,Coreference,insect repellents,Chemical
    pesticides,Chemical,Conjunction,insecticides,Chemical
    herbicides,Chemical,Coreference,pesticides,Chemical
    weed control chemicals,Chemical,Conjunction,insecticides,Chemical
    herbicides,Chemical,Coreference,weed control chemicals,Chemical
    chemical weed killers,O,Conjunction,insect repellents,Chemical
    herbicides,Chemical,Coreference,chemical weed killers,O
    insecticides,Chemical,Coreference,insect repellents,Chemical
    herbicidal agents,Chemical,Conjunction,insecticides,Chemical
    herbicides,Chemical,Coreference,herbicidal agents,Chemical
    Nutrient,Nutrient,Used_For,plant cultivation,Agri_Process
    P,O,Coreference,Nutrient,Nutrient
    crop cultivation,Agri_Process,Coreference,plant cultivation,Agri_Process
    Fertilizer,Chemical,Used_For,crop farming,Agri_Process
    P,O,Coreference,Fertilizer,Chemical
    crop cultivation,Agri_Process,Coreference,crop farming,Agri_Process
    Essential element,O,Used_For,cultivating crops,O
    P,O,Coreference,Essential element,O
    crop cultivation,Agri_Process,Coreference,cultivating crops,O
    Plant food,Crop,Used_For,crop growing,O
    P,O,Coreference,Plant food,Crop
    crop cultivation,Agri_Process,Coreference,crop growing,O
    Soil supplement,Chemical,Used_For,cultivating crops,O
    P,O,Coreference,Soil supplement,Chemical
    crop cultivation,Agri_Process,Coreference,cultivating crops,O
    soy,Crop,Conjunction,corn,Crop
    soybeans,Crop,Coreference,soy,Crop
    maize,Crop,Coreference,corn,Crop
    soy crops,Crop,Conjunction,maize,Crop
    soybeans,Crop,Coreference,soy crops,Crop
    soybean,Crop,Conjunction,corn,Crop
    soybeans,Crop,Coreference,soybean,Crop
    maize,Crop,Coreference,corn,Crop
    soy plants,Organism,Conjunction,maize,Crop
    soybeans,Crop,Coreference,soy plants,Organism
    soy cultivation,Agri_Process,Conjunction,maize,Crop
    soybeans,Crop,Coreference,soy cultivation,Agri_Process
    West Africa,Location,Origin_Of,equatorial forests,Location
    tropical forests,Location,Coreference,equatorial forests,Location
    Western Africa,Location,Origin_Of,tropical woodlands,Location
    West Africa,Location,Coreference,Western Africa,Location
    tropical forests,Location,Coreference,tropical woodlands,Location
    West African region,Location,Origin_Of,tropical jungles,Location
    West Africa,Location,Coreference,West African region,Location
    tropical forests,Location,Coreference,tropical jungles,Location
    Location in West Africa,O,Origin_Of,tropical rainforests,Location
    West Africa,Location,Coreference,Location in West Africa,O
    tropical forests,Location,Coreference,tropical rainforests,Location
    West Africa,Location,Origin_Of,tropical woodlands,Location
    tropical forests,Location,Coreference,tropical woodlands,Location
    South America,Location,Origin_Of,rainforests,Field_Area
    Latin America,Location,Coreference,South America,Location
    tropical forests,Location,Coreference,rainforests,Field_Area
    Central America,Location,Origin_Of,jungle,Location
    Latin America,Location,Coreference,Central America,Location
    tropical forests,Location,Coreference,jungle,Location
    The Americas,O,Origin_Of,tropical woodlands,Location
    Latin America,Location,Coreference,The Americas,O
    tropical forests,Location,Coreference,tropical woodlands,Location
    Southern America,Location,Origin_Of,tropical forests,Location
    Latin America,Location,Coreference,Southern America,Location
    LatAm,Location,Origin_Of,equatorial forests,Location
    Latin America,Location,Coreference,LatAm,Location
    tropical forests,Location,Coreference,equatorial forests,Location
    Asia,Location,Origin_Of,tropical jungles,Location
    tropical forests,Location,Coreference,tropical jungles,Location
    Asia,Location,Origin_Of,tropic woods,Location
    tropical forests,Location,Coreference,tropic woods,Location
    Asia,Location,Origin_Of,rainforests,Field_Area
    tropical forests,Location,Coreference,rainforests,Field_Area
    Asia,Location,Origin_Of,tropics,Location
    tropical forests,Location,Coreference,tropics,Location
    Asia,Location,Origin_Of,tropical woodlands,Location
    tropical forests,Location,Coreference,tropical woodlands,Location
    Latin American countries,Location,Conjunction,West Africa,Location
    Latin America,Location,Coreference,Latin American countries,Location
    intensification of habitat,O,Origin_Of,Eastern European,Location
    habitat intensificatio,O,Coreference,intensification of habitat,O
    process of enhancing habitats,O,Origin_Of,Eastern European,Location
    habitat intensificatio,O,Coreference,process of enhancing habitats,O
    habitat improvement,O,Origin_Of,Eastern European,Location
    habitat intensificatio,O,Coreference,habitat improvement,O
    agricultural habitat intensification,Agri_Process,Origin_Of,Eastern European,Location
    habitat intensificatio,O,Coreference,agricultural habitat intensification,Agri_Process
    habitat enhancement,O,Origin_Of,Eastern European,Location
    habitat intensificatio,O,Coreference,habitat enhancement,O
    Lesser Grey Shrike,Organism,Coreference,Avian,Location
    Birds,O,Coreference,Avian,Location
    Lesser Grey Shrike,Organism,Coreference,Flying creatures,Organism
    Birds,O,Coreference,Flying creatures,Organism
    Lesser Grey Shrike,Organism,Coreference,Feathered friends,O
    Birds,O,Coreference,Feathered friends,O
    Lesser Grey Shrike,Organism,Coreference,Winged beings,Organism
    Birds,O,Coreference,Winged beings,Organism
    Lesser Grey Shrike,Organism,Coreference,Bird species,O
    Birds,O,Coreference,Bird species,O
    Lesser Grey Shrike,Organism,Origin_Of,Western European regions,Location
    Western Europe,Location,Coreference,Western European regions,Location
    Small Grey Shrike,Organism,Origin_Of,Western Europe,Location
    Lesser Grey Shrike,Organism,Coreference,Small Grey Shrike,Organism
    Lesser Grey Shrike,Organism,Origin_Of,Western European,Location
    Lesser Grey Shrike,Organism,Origin_Of,Western Europe,Location
    National Academy of Sciences,Organization,Origin_Of,U.S.,Location
    National Research Council,Organization,Coreference,National Academy of Sciences,Organization
    National Institute of Health,Organization,Origin_Of,U.S.,Location
    National Research Council,Organization,Coreference,National Institute of Health,Organization
    National Science Foundation,Organization,Origin_Of,U.S.,Location
    National Research Council,Organization,Coreference,National Science Foundation,Organization
    "National Academies of Sciences, Engineering, and Medicine",Organization,Origin_Of,U.S.,Location
    National Research Council,Organization,Coreference,"National Academies of Sciences, Engineering, and Medicine",Organization
    NRC,Organization,Synonym_Of,National Research Council,Organization
    National Research Council,Organization,Coreference,National Research Council,Organization
    National Research Council,Organization,Synonym_Of,NRC,Organization
    NRC,Organization,Coreference,National Research Council,Organization
    National Research Council,Organization,Coreference,NRC,Organization
    Organization,O,Synonym_Of,NRC,Organization
    NRC,Organization,Coreference,Organization,O
    National Research Council,Organization,Coreference,NRC,Organization
    NRC,Organization,Synonym_Of,Research Council,Organization
    National Research Council,Organization,Coreference,Research Council,Organization
    National Research Council,Organization,Synonym_Of,Research Council,Organization
    NRC,Organization,Coreference,National Research Council,Organization
    National Research Council,Organization,Coreference,Research Council,Organization
    Microenvironment,O,Conjunction,soil moisture,Natural_Resource
    Microclimate,Agri_Process,Coreference,Microenvironment,O
    Climate microzone,O,Conjunction,soil moisture,Natural_Resource
    Microclimate,Agri_Process,Coreference,Climate microzone,O
    Local climate,O,Conjunction,soil moisture,Natural_Resource
    Microclimate,Agri_Process,Coreference,Local climate,O
    Microscopic climate,O,Conjunction,soil moisture,Natural_Resource
    Microclimate,Agri_Process,Coreference,Microscopic climate,O
    Microenvironment,O,Conjunction,soil moisture variation,Soil
    microclimate,Agri_Process,Coreference,Microenvironment,O
    soil moisture fluctuation,O,Coreference,soil moisture variation,Soil
    Microclimatic conditions,O,Conjunction,soil moisture fluctuation,O
    microclimate,Agri_Process,Coreference,Microclimatic conditions,O
    Microclimate,Agri_Process,Conjunction,soil humidity fluctuation,O
    microclimate,Agri_Process,Coreference,Microclimate,Agri_Process
    soil moisture fluctuation,O,Coreference,soil humidity fluctuation,O
    Local climate,O,Conjunction,soil moisture variation,Soil
    microclimate,Agri_Process,Coreference,Local climate,O
    soil moisture fluctuation,O,Coreference,soil moisture variation,Soil
    Microclimate,Agri_Process,Conjunction,soil moisture changes,Soil
    microclimate,Agri_Process,Coreference,Microclimate,Agri_Process
    soil moisture fluctuation,O,Coreference,soil moisture changes,Soil
    soil humidity variation,Weather,Coreference,soil moisture,Natural_Resource
    soil moisture fluctuation,O,Coreference,soil humidity variation,Weather
    soil moisture change,O,Coreference,soil moisture,Natural_Resource
    soil moisture fluctuation,O,Coreference,soil moisture change,O
    soil wetness fluctuation,O,Coreference,soil moisture,Natural_Resource
    soil moisture fluctuation,O,Coreference,soil wetness fluctuation,O
    soil moisture fluctuating,O,Coreference,soil moisture,Natural_Resource
    soil moisture fluctuation,O,Coreference,soil moisture fluctuating,O
    fluctuation in soil moisture levels,O,Coreference,soil moisture,Natural_Resource
    soil moisture fluctuation,O,Coreference,fluctuation in soil moisture levels,O
    linear regression models,ML_Model,Used_For,microclimate observations,Other_Quantity
    linear mixed models,ML_Model,Coreference,linear regression models,ML_Model
    microclimate measurements,Other_Quantity,Coreference,microclimate observations,Other_Quantity
    ML algorithms,ML_Model,Used_For,microclimate measurements,Other_Quantity
    linear mixed models,ML_Model,Coreference,ML algorithms,ML_Model
    statistical models,ML_Model,Used_For,microclimate data collection,ML_Model
    linear mixed models,ML_Model,Coreference,statistical models,ML_Model
    microclimate measurements,Other_Quantity,Coreference,microclimate data collection,ML_Model
    predictive models,ML_Model,Used_For,microclimate measurements,Other_Quantity
    linear mixed models,ML_Model,Coreference,predictive models,ML_Model
    machine learning models,ML_Model,Used_For,microclimate measurements,Other_Quantity
    linear mixed models,ML_Model,Coreference,machine learning models,ML_Model
    Agroecology,Agri_Process,Helps_In,microclimate,Agri_Process
    agroforestry,Agri_Process,Coreference,Agroecology,Agri_Process
    Agroforestry practices,Agri_Process,Helps_In,microclimate,Agri_Process
    agroforestry,Agri_Process,Coreference,Agroforestry practices,Agri_Process
    Silvopastoral systems,Agri_Process,Helps_In,microclimate,Agri_Process
    agroforestry,Agri_Process,Coreference,Silvopastoral systems,Agri_Process
    Agroforestry techniques,Agri_Process,Helps_In,microclimate,Agri_Process
    agroforestry,Agri_Process,Coreference,Agroforestry techniques,Agri_Process
    Agroforestry method,Agri_Process,Helps_In,microclimate,Agri_Process
    agroforestry,Agri_Process,Coreference,Agroforestry method,Agri_Process
    Agroforestry,Agri_Process,Helps_In,soil humidity,O
    agroforestry,Agri_Process,Coreference,Agroforestry,Agri_Process
    soil moisture,Natural_Resource,Coreference,soil humidity,O
    Agri_Process of cultivating trees,Agri_Method,Helps_In,soil moisture,Natural_Resource
    agroforestry,Agri_Process,Coreference,Agri_Process of cultivating trees,Agri_Method
    Agroforestry technique,Agri_Process,Helps_In,soil wetness,O
    agroforestry,Agri_Process,Coreference,Agroforestry technique,Agri_Process
    soil moisture,Natural_Resource,Coreference,soil wetness,O
    Agroforestry practice,Agri_Process,Helps_In,soil dampness,O
    agroforestry,Agri_Process,Coreference,Agroforestry practice,Agri_Process
    soil moisture,Natural_Resource,Coreference,soil dampness,O
    Agri_Process incorporating trees,Agri_Method,Helps_In,soil moisture,Natural_Resource
    agroforestry,Agri_Process,Coreference,Agri_Process incorporating trees,Agri_Method
    revenue,O,Coreference,turnover,O
    irrigation techniques,O,Conjunction,methods of crop cultivation,O
    irrigation methods,O,Coreference,irrigation techniques,O
    methods of crop farming,O,Coreference,methods of crop cultivation,O
    irrigation practices,O,Conjunction,methods of farming crops,O
    irrigation methods,O,Coreference,irrigation practices,O
    methods of crop farming,O,Coreference,methods of farming crops,O
    irrigation systems,Agri_Process,Conjunction,ways of cultivating crops,O
    irrigation methods,O,Coreference,irrigation systems,Agri_Process
    methods of crop farming,O,Coreference,ways of cultivating crops,O
    irrigation approaches,O,Conjunction,techniques of planting crops,O
    irrigation methods,O,Coreference,irrigation approaches,O
    methods of crop farming,O,Coreference,techniques of planting crops,O
    Crop cultivation techniques,Agri_Method,Conjunction,adapted cropping pattern,O
    methods of crop farming,O,Coreference,Crop cultivation techniques,Agri_Method
    Ways of agricultural cultivation,O,Conjunction,adapted cropping pattern,O
    methods of crop farming,O,Coreference,Ways of agricultural cultivation,O
    Agri_Methods,Agri_Method,Conjunction,adapted cropping pattern,O
    methods of crop farming,O,Coreference,Agri_Methods,Agri_Method
    Techniques for crop farming,O,Conjunction,adapted cropping pattern,O
    methods of crop farming,O,Coreference,Techniques for crop farming,O
    Agri_Methodologies,ML_Model,Conjunction,adapted cropping pattern,O
    methods of crop farming,O,Coreference,Agri_Methodologies,ML_Model
    Intensive farming practices,O,Conjunction,Lesser Grey Shrike,Organism
    agricultural intensifiation,Agri_Process,Coreference,Intensive farming practices,O
    Greenhouse gas,Nutrient,Synonym_Of,greenhouse gases,Chemical
    GHG,Location,Coreference,Greenhouse gas,Nutrient
    Emission gas,O,Synonym_Of,greenhouse gases,Chemical
    GHG,Location,Coreference,Emission gas,O
    Climate gas,Natural_Disaster,Synonym_Of,greenhouse gases,Chemical
    GHG,Location,Coreference,Climate gas,Natural_Disaster
    Global warming gas,Natural_Disaster,Synonym_Of,greenhouse gases,Chemical
    GHG,Location,Coreference,Global warming gas,Natural_Disaster
    Carbon dioxide equivalents,Chemical,Synonym_Of,greenhouse gases,Chemical
    GHG,Location,Coreference,Carbon dioxide equivalents,Chemical
    Carbon dioxide,Chemical,Synonym_Of,carbonic acid,Chemical
    CO2,Chemical,Coreference,Carbon dioxide,Chemical
    carbon dioxide,Chemical,Coreference,carbonic acid,Chemical
    Carbon dioxide gas,Chemical,Synonym_Of,carbonic anhydride,Chemical
    CO2,Chemical,Coreference,Carbon dioxide gas,Chemical
    carbon dioxide,Chemical,Coreference,carbonic anhydride,Chemical
    Carbon dioxide compound,Chemical,Synonym_Of,carbonic acid gas,Chemical
    CO2,Chemical,Coreference,Carbon dioxide compound,Chemical
    carbon dioxide,Chemical,Coreference,carbonic acid gas,Chemical
    CO2 molecule,Chemical,Synonym_Of,carbon dioxide gas,Chemical
    CO2,Chemical,Coreference,CO2 molecule,Chemical
    carbon dioxide,Chemical,Coreference,carbon dioxide gas,Chemical
    Nitrogen oxide,Nutrient,Synonym_Of,nitrous oxide,Chemical
    N2O,Chemical,Coreference,Nitrogen oxide,Nutrient
    Dinitrogen monoxide,Chemical,Synonym_Of,nitrous oxide,Chemical
    N2O,Chemical,Coreference,Dinitrogen monoxide,Chemical
    Laughing gas,Disease,Synonym_Of,nitrous oxide,Chemical
    N2O,Chemical,Coreference,Laughing gas,Disease
    N2O,Chemical,Synonym_Of,dinitrogen monoxide,Chemical
    nitrous oxide,Chemical,Coreference,dinitrogen monoxide,Chemical
    Nitrogen (II) oxide,Nutrient,Synonym_Of,nitrous oxide,Chemical
    N2O,Chemical,Coreference,Nitrogen (II) oxide,Nutrient
    Greenhouse emissions,Chemical,Includes,CO2,Chemical
    greenhouse gases,Chemical,Coreference,Greenhouse emissions,Chemical
    carbon dioxide,Chemical,Coreference,CO2,Chemical
    Gaseous pollutants,Chemical,Includes,carbon dioxide,Chemical
    greenhouse gases,Chemical,Coreference,Gaseous pollutants,Chemical
    Emission gases,O,Includes,CO2,Chemical
    greenhouse gases,Chemical,Coreference,Emission gases,O
    carbon dioxide,Chemical,Coreference,CO2,Chemical
    Atmospheric pollutants,Agri_Pollution,Includes,carbon dioxide,Chemical
    greenhouse gases,Chemical,Coreference,Atmospheric pollutants,Agri_Pollution
    Greenhouse vapors,Chemical,Includes,carbon dioxide,Chemical
    greenhouse gases,Chemical,Coreference,Greenhouse vapors,Chemical
    Greenhouse emissions,Chemical,Includes,nitrous oxide,Chemical
    greenhouse gases,Chemical,Coreference,Greenhouse emissions,Chemical
    Greenhouse chemicals,Chemical,Includes,nitrous oxide,Chemical
    greenhouse gases,Chemical,Coreference,Greenhouse chemicals,Chemical
    Greenhouse substances,Chemical,Includes,nitrous oxide,Chemical
    greenhouse gases,Chemical,Coreference,Greenhouse substances,Chemical
    Greenhouse compounds,Nutrient,Includes,nitrous oxide,Chemical
    greenhouse gases,Chemical,Coreference,Greenhouse compounds,Nutrient
    methane gas,Chemical,Conjunction,nitrous oxide,Chemical
    methane,Nutrient,Coreference,methane gas,Chemical
    natural gas,Natural_Resource,Conjunction,nitrous oxide,Chemical
    methane,Nutrient,Coreference,natural gas,Natural_Resource
    Global warming,Natural_Disaster,Includes,rise in ocean level,O
    Climate change,Natural_Disaster,Coreference,Global warming,Natural_Disaster
    rise in sea level,O,Coreference,rise in ocean level,O
    Climate crisis,Natural_Disaster,Includes,rise in sea level,O
    Climate change,Natural_Disaster,Coreference,Climate crisis,Natural_Disaster
    Environmental changes,O,Includes,rise in sea levels,O
    Climate change,Natural_Disaster,Coreference,Environmental changes,O
    rise in sea level,O,Coreference,rise in sea levels,O
    Weather shift,O,Includes,rise in ocean level,O
    Climate change,Natural_Disaster,Coreference,Weather shift,O
    rise in sea level,O,Coreference,rise in ocean level,O
    Ecological imbalance,O,Includes,increase in sea level,O
    Climate change,Natural_Disaster,Coreference,Ecological imbalance,O
    rise in sea level,O,Coreference,increase in sea level,O
    Global warming,Natural_Disaster,Includes,changes in rainfall patterns,O
    Climate change,Natural_Disaster,Coreference,Global warming,Natural_Disaster
    changes in rainfall sequences,O,Coreference,changes in rainfall patterns,O
    Environmental change,O,Includes,changes in precipitation patterns,O
    Climate change,Natural_Disaster,Coreference,Environmental change,O
    changes in rainfall sequences,O,Coreference,changes in precipitation patterns,O
    Weather shifts,O,Includes,changes in rainfall sequences,O
    Climate change,Natural_Disaster,Coreference,Weather shifts,O
    Natural calamity,O,Includes,changes in rainfall fluctuations,O
    Climate change,Natural_Disaster,Coreference,Natural calamity,O
    changes in rainfall sequences,O,Coreference,changes in rainfall fluctuations,O
    Climate variability,O,Includes,alterations in rainfall sequences,O
    Climate change,Natural_Disaster,Coreference,Climate variability,O
    changes in rainfall sequences,O,Coreference,alterations in rainfall sequences,O
    Climate variability,O,Includes,movement of climatic regions,O
    Climate change,Natural_Disaster,Coreference,Climate variability,O
    Global warming,Natural_Disaster,Includes,movement of climatic regions,O
    Climate change,Natural_Disaster,Coreference,Global warming,Natural_Disaster
    Environmental changes,O,Includes,movement of climatic regions,O
    Climate change,Natural_Disaster,Coreference,Environmental changes,O
    Natural calamities,O,Includes,movement of climatic regions,O
    Climate change,Natural_Disaster,Coreference,Natural calamities,O
    Weather shifts,O,Includes,movement of climatic regions,O
    Climate change,Natural_Disaster,Coreference,Weather shifts,O
    Rising temperatures,O,Caused_By,shift of climatic zones,O
    increased temperatures,O,Coreference,Rising temperatures,O
    movement of climatic regions,O,Coreference,shift of climatic zones,O
    Elevated temperatures,O,Caused_By,migration of climatic areas,O
    increased temperatures,O,Coreference,Elevated temperatures,O
    movement of climatic regions,O,Coreference,migration of climatic areas,O
    Higher temperatures,O,Caused_By,movement of climate regions,O
    increased temperatures,O,Coreference,Higher temperatures,O
    movement of climatic regions,O,Coreference,movement of climate regions,O
    Warmer temperatures,O,Caused_By,transition of climate zones,O
    increased temperatures,O,Coreference,Warmer temperatures,O
    movement of climatic regions,O,Coreference,transition of climate zones,O
    Escalated temperatures,O,Caused_By,displacement of climatic regions,O
    increased temperatures,O,Coreference,Escalated temperatures,O
    movement of climatic regions,O,Coreference,displacement of climatic regions,O
    Variations in precipitation patterns,O,Conjunction,movement of climatic regions,O
    changes in rainfall sequences,O,Coreference,Variations in precipitation patterns,O
    Shifts in rainfall sequences,O,Conjunction,movement of climatic regions,O
    changes in rainfall sequences,O,Coreference,Shifts in rainfall sequences,O
    Fluctuations in rainfall sequences,O,Conjunction,movement of climatic regions,O
    changes in rainfall sequences,O,Coreference,Fluctuations in rainfall sequences,O
    Alterations in precipitation sequences,O,Conjunction,movement of climatic regions,O
    changes in rainfall sequences,O,Coreference,Alterations in precipitation sequences,O
    Modifications in rainfall patterns,O,Conjunction,movement of climatic regions,O
    changes in rainfall sequences,O,Coreference,Modifications in rainfall patterns,O
    drought conditions,O,Conjunction,tempests,O
    droughts,Natural_Disaster,Coreference,drought conditions,O
    storms,O,Coreference,tempests,O
    water shortages,O,Conjunction,thunderstorms,Natural_Disaster
    droughts,Natural_Disaster,Coreference,water shortages,O
    storms,O,Coreference,thunderstorms,Natural_Disaster
    arid periods,O,Conjunction,hurricanes,Natural_Disaster
    droughts,Natural_Disaster,Coreference,arid periods,O
    storms,O,Coreference,hurricanes,Natural_Disaster
    rainfall deficits,O,Conjunction,cyclones,Natural_Disaster
    droughts,Natural_Disaster,Coreference,rainfall deficits,O
    storms,O,Coreference,cyclones,Natural_Disaster
    tornadoes,Natural_Disaster,Conjunction,floods,Natural_Disaster
    storms,O,Coreference,tornadoes,Natural_Disaster
    flood,Natural_Disaster,Coreference,floods,Natural_Disaster
    cyclones,Natural_Disaster,Conjunction,flood,Natural_Disaster
    storms,O,Coreference,cyclones,Natural_Disaster
    hurricanes,Natural_Disaster,Conjunction,flooding,O
    storms,O,Coreference,hurricanes,Natural_Disaster
    flood,Natural_Disaster,Coreference,flooding,O
    thunderstorms,Natural_Disaster,Conjunction,floods,Natural_Disaster
    storms,O,Coreference,thunderstorms,Natural_Disaster
    flood,Natural_Disaster,Coreference,floods,Natural_Disaster
    typhoons,Natural_Disaster,Conjunction,flood,Natural_Disaster
    storms,O,Coreference,typhoons,Natural_Disaster
    Natural disaster,O,Caused_By,fluctuating climatic patterns,O
    flood,Natural_Disaster,Coreference,Natural disaster,O
    changing climatic patterns,O,Coreference,fluctuating climatic patterns,O
    Flood,Natural_Disaster,Caused_By,shifting climate trends,O
    flood,Natural_Disaster,Coreference,Flood,Natural_Disaster
    changing climatic patterns,O,Coreference,shifting climate trends,O
    Natural calamity,O,Caused_By,changing climatic conditions,O
    flood,Natural_Disaster,Coreference,Natural calamity,O
    changing climatic patterns,O,Coreference,changing climatic conditions,O
    Flood,Natural_Disaster,Caused_By,evolving climate variations,O
    flood,Natural_Disaster,Coreference,Flood,Natural_Disaster
    changing climatic patterns,O,Coreference,evolving climate variations,O
    Catastrophic flood,O,Caused_By,varying weather patterns,O
    flood,Natural_Disaster,Coreference,Catastrophic flood,O
    changing climatic patterns,O,Coreference,varying weather patterns,O
    cyclones,Natural_Disaster,Caused_By,changing climatic patterns,O
    storms,O,Coreference,cyclones,Natural_Disaster
    hurricanes,Natural_Disaster,Caused_By,changing climatic patterns,O
    storms,O,Coreference,hurricanes,Natural_Disaster
    typhoons,Natural_Disaster,Caused_By,changing climatic patterns,O
    storms,O,Coreference,typhoons,Natural_Disaster
    tornadoes,Natural_Disaster,Caused_By,changing climatic patterns,O
    storms,O,Coreference,tornadoes,Natural_Disaster
    tempests,O,Caused_By,changing climatic patterns,O
    storms,O,Coreference,tempests,O
    dry spells,O,Caused_By,shifting weather patterns,O
    droughts,Natural_Disaster,Coreference,dry spells,O
    changing climatic patterns,O,Coreference,shifting weather patterns,O
    water shortages,O,Caused_By,evolving climate conditions,O
    droughts,Natural_Disaster,Coreference,water shortages,O
    changing climatic patterns,O,Coreference,evolving climate conditions,O
    arid conditions,O,Caused_By,fluctuating atmospheric trends,O
    droughts,Natural_Disaster,Coreference,arid conditions,O
    changing climatic patterns,O,Coreference,fluctuating atmospheric trends,O
    lack of precipitation,O,Caused_By,varying climatic cycles,O
    droughts,Natural_Disaster,Coreference,lack of precipitation,O
    changing climatic patterns,O,Coreference,varying climatic cycles,O
    dehydration events,O,Caused_By,altering meteorological systems,O
    droughts,Natural_Disaster,Coreference,dehydration events,O
    changing climatic patterns,O,Coreference,altering meteorological systems,O
    Shifting weather patterns,O,Coreference,rising temperatures,O
    changing climatic patterns,O,Coreference,Shifting weather patterns,O
    increased temperatures,O,Coreference,rising temperatures,O
    Fluctuating climate trends,O,Coreference,escalating temperatures,O
    changing climatic patterns,O,Coreference,Fluctuating climate trends,O
    increased temperatures,O,Coreference,escalating temperatures,O
    Altering climatic conditions,O,Coreference,higher temperatures,O
    changing climatic patterns,O,Coreference,Altering climatic conditions,O
    increased temperatures,O,Coreference,higher temperatures,O
    Varying weather cycles,O,Coreference,elevated temperatures,O
    changing climatic patterns,O,Coreference,Varying weather cycles,O
    increased temperatures,O,Coreference,elevated temperatures,O
    Evolution of climate patterns,O,Coreference,growing temperatures,O
    changing climatic patterns,O,Coreference,Evolution of climate patterns,O
    increased temperatures,O,Coreference,growing temperatures,O
    Carbon dioxide,Chemical,Caused_By,Human activities,O
    CO2,Chemical,Coreference,Carbon dioxide,Chemical
    Human,O,Coreference,Human activities,O
    CO2,Chemical,Caused_By,Humans,Location
    Human,O,Coreference,Humans,Location
    Carbon dioxide,Chemical,Caused_By,Human influence,O
    CO2,Chemical,Coreference,Carbon dioxide,Chemical
    Human,O,Coreference,Human influence,O
    Carbon dioxide,Chemical,Caused_By,Human impact,O
    CO2,Chemical,Coreference,Carbon dioxide,Chemical
    Human,O,Coreference,Human impact,O
    Carbon dioxide,Chemical,Caused_By,Human input,O
    CO2,Chemical,Coreference,Carbon dioxide,Chemical
    Human,O,Coreference,Human input,O
    Carbon Dioxide,Chemical,Caused_By,coal,O
    CO2,Chemical,Coreference,Carbon Dioxide,Chemical
    fossil fuels,Natural_Resource,Coreference,coal,O
    CO2,Chemical,Caused_By,petroleum products,Chemical
    fossil fuels,Natural_Resource,Coreference,petroleum products,Chemical
    Carbon Dioxide,Chemical,Caused_By,natural gas,Natural_Resource
    CO2,Chemical,Coreference,Carbon Dioxide,Chemical
    fossil fuels,Natural_Resource,Coreference,natural gas,Natural_Resource
    Carbon Dioxide,Chemical,Caused_By,hydrocarbons,Nutrient
    CO2,Chemical,Coreference,Carbon Dioxide,Chemical
    fossil fuels,Natural_Resource,Coreference,hydrocarbons,Nutrient
    CO2,Chemical,Caused_By,fossil energy sources,O
    fossil fuels,Natural_Resource,Coreference,fossil energy sources,O
    Agri_Process,ML_Model,Conjunction,water or labor saving technologies,Natural_Resource
    variance-reducing,O,Coreference,Agri_Process,ML_Model
    water or labor reducing technologies,Natural_Resource,Coreference,water or labor saving technologies,Natural_Resource
    Variance-decreasing,O,Conjunction,water or labor reducing technologies,Natural_Resource
    variance-reducing,O,Coreference,Variance-decreasing,O
    Agri_Process,ML_Model,Conjunction,water or labor minimizing technologies,Natural_Resource
    variance-reducing,O,Coreference,Agri_Process,ML_Model
    water or labor reducing technologies,Natural_Resource,Coreference,water or labor minimizing technologies,Natural_Resource
    Variance-reducing,O,Conjunction,water or labor conserving technologies,Natural_Resource
    variance-reducing,O,Coreference,Variance-reducing,O
    water or labor reducing technologies,Natural_Resource,Coreference,water or labor conserving technologies,Natural_Resource
    Agri_Process,ML_Model,Conjunction,water or labor cutting technologies,Natural_Resource
    variance-reducing,O,Coreference,Agri_Process,ML_Model
    water or labor reducing technologies,Natural_Resource,Coreference,water or labor cutting technologies,Natural_Resource
    New Zealand,Location,Synonym_Of,Aotearoa,Location
    NZ,Location,Coreference,New Zealand,Location
    New Zealand,Location,Coreference,Aotearoa,Location
    NZ,Location,Synonym_Of,Aotearoa,Location
    New Zealand,Location,Coreference,Aotearoa,Location
    New Zealand,Location,Synonym_Of,Land of the Long White Cloud,Location
    NZ,Location,Coreference,New Zealand,Location
    New Zealand,Location,Coreference,Land of the Long White Cloud,Location
    NZ,Location,Synonym_Of,Land of the Long White Cloud,Location
    New Zealand,Location,Coreference,Land of the Long White Cloud,Location
    New Zealand,Location,Synonym_Of,Kiwi land,Location
    NZ,Location,Coreference,New Zealand,Location
    New Zealand,Location,Coreference,Kiwi land,Location
    Senior et al 2013,Citation,Conjunction,Edwards et al 2013,Citation
    Edwards et al 2013,Citation,Coreference,Edwards et al 2013,Citation
    Carbon Dioxide,Chemical,Caused_By,clear-cutting,O
    CO2,Chemical,Coreference,Carbon Dioxide,Chemical
    deforestation,O,Coreference,clear-cutting,O
    CO2,Chemical,Caused_By,forest degradation,O
    deforestation,O,Coreference,forest degradation,O
    Carbon Dioxide,Chemical,Caused_By,logging activities,O
    CO2,Chemical,Coreference,Carbon Dioxide,Chemical
    deforestation,O,Coreference,logging activities,O
    Complete farm planning toolkits,O,Conjunction,integrated pest control,O
    whole farm planning toolkits,O,Coreference,Complete farm planning toolkits,O
    integrated pest management,O,Coreference,integrated pest control,O
    Entire farm planning toolkits,O,Conjunction,integrated pest management,O
    whole farm planning toolkits,O,Coreference,Entire farm planning toolkits,O
    Full farm planning toolkits,O,Conjunction,integrated pest management,O
    whole farm planning toolkits,O,Coreference,Full farm planning toolkits,O
    Whole-farm planning toolkits,Technology,Conjunction,integrated pest control,O
    whole farm planning toolkits,O,Coreference,Whole-farm planning toolkits,Technology
    integrated pest management,O,Coreference,integrated pest control,O
    Total farm planning toolkits,O,Conjunction,integrated pest management,O
    whole farm planning toolkits,O,Coreference,Total farm planning toolkits,O
    plantation,O,Conjunction,orchard,Field_Area
    farm,O,Coreference,plantation,O
    ranch,O,Conjunction,orchard,Field_Area
    farm,O,Coreference,ranch,O
    homestead,O,Conjunction,orchard,Field_Area
    farm,O,Coreference,homestead,O
    plantation,O,Conjunction,woodland,Location
    orchard,Field_Area,Coreference,plantation,O
    forest,O,Coreference,woodland,Location
    Agricultural Institution,Organization,Synonym_Of,Agricultural Innovation Systems,Organization
    AIS,Location,Coreference,Agricultural Institution,Organization
    Agricultural Organization,Organization,Synonym_Of,Agricultural Innovation Systems,Organization
    AIS,Location,Coreference,Agricultural Organization,Organization
    Farming Group,Person,Synonym_Of,Agricultural Innovation Systems,Organization
    AIS,Location,Coreference,Farming Group,Person
    Crop Management Firm,Organization,Synonym_Of,Agricultural Innovation Systems,Organization
    AIS,Location,Coreference,Crop Management Firm,Organization
    Multi-stakeholder engagement,O,Helps_In,farmers,Person
    multi-stakeholders approach,O,Coreference,Multi-stakeholder engagement,O
    Collaborative approach,O,Helps_In,farmers,Person
    multi-stakeholders approach,O,Coreference,Collaborative approach,O
    Partnership strategy,O,Helps_In,farmers,Person
    multi-stakeholders approach,O,Coreference,Partnership strategy,O
    Cooperative method,O,Helps_In,farmers,Person
    multi-stakeholders approach,O,Coreference,Cooperative method,O
    Participatory model,O,Helps_In,farmers,Person
    multi-stakeholders approach,O,Coreference,Participatory model,O
    Farmers,Person,Helps_In,land management,O
    arm management,O,Coreference,land management,O
    Farmers,Person,Helps_In,field management,O
    arm management,O,Coreference,field management,O
    Farmers,Person,Helps_In,agricultural land management,Agri_Process
    arm management,O,Coreference,agricultural land management,Agri_Process
    Farmers,Person,Helps_In,cultivation management,O
    arm management,O,Coreference,cultivation management,O
    Farmers,Person,Helps_In,farming management,O
    arm management,O,Coreference,farming management,O
    T&V,O,Origin_Of,India,Location
    Training and Visit (T&V) extension,Agri_Method,Coreference,T&V,O
    Training and Visit (T&V) program,Technology,Origin_Of,India,Location
    Training and Visit (T&V) extension,Agri_Method,Coreference,Training and Visit (T&V) program,Technology
    Extension services,O,Origin_Of,India,Location
    Training and Visit (T&V) extension,Agri_Method,Coreference,Extension services,O
    Training and Visit (T&V) system,Technology,Origin_Of,India,Location
    Training and Visit (T&V) extension,Agri_Method,Coreference,Training and Visit (T&V) system,Technology
    T&V approach,O,Origin_Of,India,Location
    Training and Visit (T&V) extension,Agri_Method,Coreference,T&V approach,O
    Agricultural knowledge and innovation systems,O,Helps_In,Farmer Field School and Landcare,Organization
    Farming knowledge and innovation systems,O,Helps_In,Farmer Field School and Landcare,Organization
    Agribusiness expertise and innovation networks,O,Helps_In,Farmer Field School and Landcare,Organization
    Organizational agricultural knowledge and innovation systems,O,Helps_In,Farmer Field School and Landcare,Organization
    agricultural knowledge and innovation systems,O,Coreference,Organizational agricultural knowledge and innovation systems,O
    Rural development knowledge and innovation systems,O,Helps_In,Farmer Field School and Landcare,Organization
    Northeast area,Location,Includes,Czech Republic,Location
    North Eastern region,Location,Coreference,Northeast area,Location
    Eastern region,Location,Includes,Czech Republic,Location
    North Eastern region,Location,Coreference,Eastern region,Location
    Area in the Northeast,O,Includes,Czech Republic,Location
    North Eastern region,Location,Coreference,Area in the Northeast,O
    North East region,Location,Includes,Czech Republic,Location
    North Eastern region,Location,Coreference,North East region,Location
    Eastern territory,Location,Includes,Czech Republic,Location
    North Eastern region,Location,Coreference,Eastern territory,Location
    Northern Eastern area,Location,Includes,Slovak Republic,Location
    North Eastern region,Location,Coreference,Northern Eastern area,Location
    Northeast region,Location,Includes,Slovak Republic,Location
    North Eastern region,Location,Coreference,Northeast region,Location
    North East territory,Location,Includes,Slovak Republic,Location
    North Eastern region,Location,Coreference,North East territory,Location
    Region in the Northeast,O,Includes,Slovak Republic,Location
    North Eastern region,Location,Coreference,Region in the Northeast,O
    North Eastern zone,Location,Includes,Slovak Republic,Location
    North Eastern region,Location,Coreference,North Eastern zone,Location
    North East area,Location,Includes,Poland,Location
    North Eastern region,Location,Coreference,North East area,Location
    Region in the North East,O,Includes,Poland,Location
    North Eastern region,Location,Coreference,Region in the North East,O
    North Eastern territory,Location,Includes,Poland,Location
    North Eastern region,Location,Coreference,North Eastern territory,Location
    Northeastern zone,Location,Includes,Poland,Location
    North Eastern region,Location,Coreference,Northeastern zone,Location
    Area in the Northeast,O,Includes,Poland,Location
    North Eastern region,Location,Coreference,Area in the Northeast,O
    Southern part,O,Includes,Bulgaria,Location
    South Eastern region,Location,Coreference,Southern part,O
    Southeastern area,Location,Includes,Bulgaria,Location
    South Eastern region,Location,Coreference,Southeastern area,Location
    Region in the southeast,O,Includes,Bulgaria,Location
    South Eastern region,Location,Coreference,Region in the southeast,O
    South East zone,Location,Includes,Bulgaria,Location
    South Eastern region,Location,Coreference,South East zone,Location
    Balkan Peninsula region,Location,Includes,Bulgaria,Location
    South Eastern region,Location,Coreference,Balkan Peninsula region,Location
    South East area,Location,Includes,Hungary,Location
    South Eastern region,Location,Coreference,South East area,Location
    South Eastern territory,Location,Includes,Hungary,Location
    South Eastern region,Location,Coreference,South Eastern territory,Location
    Southeastern zone,Location,Includes,Hungary,Location
    South Eastern region,Location,Coreference,Southeastern zone,Location
    South Eastern part,Location,Includes,Hungary,Location
    South Eastern region,Location,Coreference,South Eastern part,Location
    South East district,Location,Includes,Hungary,Location
    South Eastern region,Location,Coreference,South East district,Location
    South Eastern area,Location,Includes,Romania,Location
    South Eastern region,Location,Coreference,South Eastern area,Location
    South East territory,Location,Includes,Romania,Location
    South Eastern region,Location,Coreference,South East territory,Location
    South Eastern zone,Location,Includes,Romania,Location
    South Eastern region,Location,Coreference,South Eastern zone,Location
    Region in the Southeast,O,Includes,Romania,Location
    South Eastern region,Location,Coreference,Region in the Southeast,O
    South Eastern sector,Location,Includes,Romania,Location
    South Eastern region,Location,Coreference,South Eastern sector,Location
    Southeastern area,Location,Includes,Bosnia,Location
    South Eastern region,Location,Coreference,Southeastern area,Location
    South Eastern territory,Location,Includes,Bosnia,Location
    South Eastern region,Location,Coreference,South Eastern territory,Location
    Southeastern region,Location,Includes,Bosnia,Location
    South Eastern region,Location,Coreference,Southeastern region,Location
    South Eastern zone,Location,Includes,Bosnia,Location
    South Eastern region,Location,Coreference,South Eastern zone,Location
    South Eastern district,Location,Includes,Bosnia,Location
    South Eastern region,Location,Coreference,South Eastern district,Location
    South Eastern area,Location,Includes,Herzegovina,Location
    South Eastern region,Location,Coreference,South Eastern area,Location
    South Eastern territory,Location,Includes,Herzegovina,Location
    South Eastern region,Location,Coreference,South Eastern territory,Location
    South East zone,Location,Includes,Herzegovina,Location
    South Eastern region,Location,Coreference,South East zone,Location
    South Eastern district,Location,Includes,Herzegovina,Location
    South Eastern region,Location,Coreference,South Eastern district,Location
    South East region,Location,Includes,Herzegovina,Location
    South Eastern region,Location,Coreference,South East region,Location
    South East area,Location,Includes,Croatia,Location
    South Eastern region,Location,Coreference,South East area,Location
    Southeast territory,Location,Includes,Croatia,Location
    South Eastern region,Location,Coreference,Southeast territory,Location
    South Eastern district,Location,Includes,Croatia,Location
    South Eastern region,Location,Coreference,South Eastern district,Location
    Area in South East,O,Includes,Croatia,Location
    South Eastern region,Location,Coreference,Area in South East,O
    South Eastern part,Location,Includes,Croatia,Location
    South Eastern region,Location,Coreference,South Eastern part,Location
    South East area,Location,Includes,Macedonia,Location
    South Eastern region,Location,Coreference,South East area,Location
    South Eastern territory,Location,Includes,Macedonia,Location
    South Eastern region,Location,Coreference,South Eastern territory,Location
    South Eastern zone,Location,Includes,Macedonia,Location
    South Eastern region,Location,Coreference,South Eastern zone,Location
    South East region,Location,Includes,Macedonia,Location
    South Eastern region,Location,Coreference,South East region,Location
    South Easterly region,Location,Includes,Macedonia,Location
    South Eastern region,Location,Coreference,South Easterly region,Location
    South East area,Location,Includes,Slovenia,Location
    South Eastern region,Location,Coreference,South East area,Location
    Southeastern zone,Location,Includes,Slovenia,Location
    South Eastern region,Location,Coreference,Southeastern zone,Location
    South Eastern territory,Location,Includes,Slovenia,Location
    South Eastern region,Location,Coreference,South Eastern territory,Location
    Southeastern region,Location,Includes,Slovenia,Location
    South Eastern region,Location,Coreference,Southeastern region,Location
    South East area,Location,Includes,Serbia,Location
    South Eastern region,Location,Coreference,South East area,Location
    Southeast territory,Location,Includes,Serbia,Location
    South Eastern region,Location,Coreference,Southeast territory,Location
    South Eastern zone,Location,Includes,Serbia,Location
    South Eastern region,Location,Coreference,South Eastern zone,Location
    South East region,Location,Includes,Serbia,Location
    South Eastern region,Location,Coreference,South East region,Location
    South Eastern district,Location,Includes,Serbia,Location
    South Eastern region,Location,Coreference,South Eastern district,Location
    South East area,Location,Includes,Montenegro,Location
    South Eastern region,Location,Coreference,South East area,Location
    Southeastern territory,Location,Includes,Montenegro,Location
    South Eastern region,Location,Coreference,Southeastern territory,Location
    Region of Southeast,Location,Includes,Montenegro,Location
    South Eastern region,Location,Coreference,Region of Southeast,Location
    Southeastern zone,Location,Includes,Montenegro,Location
    South Eastern region,Location,Coreference,Southeastern zone,Location
    South Eastern district,Location,Includes,Montenegro,Location
    South Eastern region,Location,Coreference,South Eastern district,Location
    Sudan Savannah,Location,Includes,northeastern portion of Upper East region,Location
    north-eastern part of Upper East region,Location,Coreference,northeastern portion of Upper East region,Location
    Sudan Savannah,Location,Includes,northeastern area of Upper East region,Location
    north-eastern part of Upper East region,Location,Coreference,northeastern area of Upper East region,Location
    Sudanian Savannah,Location,Includes,north-eastern part of Upper East region,Location
    Sudan Savannah,Location,Coreference,Sudanian Savannah,Location
    Sudan Savannah,Location,Includes,north-eastern section of Upper East region,Location
    Sudanian Savannah,Location,Includes,northeastern part of Upper East region,Location
    Toxic elements,Nutrient,Includes,Pb,Chemical
    Heavy metals,Crop,Coreference,Toxic elements,Nutrient
    Metallic pollutants,Chemical,Includes,Pb,Chemical
    Heavy metals,Crop,Coreference,Metallic pollutants,Chemical
    Hazardous metals,O,Includes,Pb,Chemical
    Heavy metals,Crop,Coreference,Hazardous metals,O
    Pollution agents,Person,Includes,Pb,Chemical
    Heavy metals,Crop,Coreference,Pollution agents,Person
    Contaminants,Chemical,Includes,Pb,Chemical
    Heavy metals,Crop,Coreference,Contaminants,Chemical
    Toxic elements,Nutrient,Includes,Cu,Location
    Heavy metals,Crop,Coreference,Toxic elements,Nutrient
    Metallic pollutants,Chemical,Includes,Cu,Location
    Heavy metals,Crop,Coreference,Metallic pollutants,Chemical
    Hazardous metals,O,Includes,Cu,Location
    Heavy metals,Crop,Coreference,Hazardous metals,O
    Contaminants,Chemical,Includes,Cu,Location
    Heavy metals,Crop,Coreference,Contaminants,Chemical
    Metallic substances,O,Includes,Cu,Location
    Heavy metals,Crop,Coreference,Metallic substances,O
    Metallic elements,O,Includes,Fe,O
    Heavy metals,Crop,Coreference,Metallic elements,O
    Metallic compounds,O,Includes,Fe,O
    Heavy metals,Crop,Coreference,Metallic compounds,O
    Trace metals,O,Includes,Fe,O
    Heavy metals,Crop,Coreference,Trace metals,O
    Toxic elements,Nutrient,Includes,Zinc,Crop
    Heavy metals,Crop,Coreference,Toxic elements,Nutrient
    Zn,O,Coreference,Zinc,Crop
    Metallic pollutants,Chemical,Includes,Zn,O
    Heavy metals,Crop,Coreference,Metallic pollutants,Chemical
    Watermelon,Fruit,Seasonal,summer,Season
    summer,Season,Coreference,summer,Season
    Watermelon,Fruit,Seasonal,summertime,Season
    summer,Season,Coreference,summertime,Season
    Watermelon,Fruit,Seasonal,warm season,Season
    summer,Season,Coreference,warm season,Season
    Watermelon,Fruit,Seasonal,sunny period,O
    summer,Season,Coreference,sunny period,O
    Watermelon,Fruit,Seasonal,hot season,Season
    summer,Season,Coreference,hot season,Season
    watermelon,Fruit,Seasonal,Summer,Season
    Summer,Season,Coreference,Summer,Season
    watermelon,Fruit,Seasonal,Hot season,Season
    Summer,Season,Coreference,Hot season,Season
    watermelon,Fruit,Seasonal,Warm period,Season
    Summer,Season,Coreference,Warm period,Season
    fruit,O,Seasonal,Summer,Season
    watermelon,Fruit,Coreference,fruit,O
    fruit,O,Seasonal,Warm season,Season
    watermelon,Fruit,Coreference,fruit,O
    Summer,Season,Coreference,Warm season,Season
    Fruit,O,Includes,Kesar mangoes,Fruit
    Mango,Fruit,Coreference,Fruit,O
    Mangoes,Crop,Includes,Kesar mangoes,Fruit
    Mango,Fruit,Coreference,Mangoes,Crop
    Tropical fruit,Fruit,Includes,Kesar mangoes,Fruit
    Mango,Fruit,Coreference,Tropical fruit,Fruit
    Mango variety,Fruit,Includes,Kesar mangoes,Fruit
    Mango,Fruit,Coreference,Mango variety,Fruit
    Exotic fruit,O,Includes,Kesar mangoes,Fruit
    Mango,Fruit,Coreference,Exotic fruit,O
    Mango,Fruit,Includes,Langra,Location
    Himsagar,Location,Coreference,Langra,Location
    Mango fruit,Fruit,Includes,Himsagar,Location
    Mango,Fruit,Coreference,Mango fruit,Fruit
    Mangoes,Crop,Includes,Himsagar,Location
    Mango,Fruit,Coreference,Mangoes,Crop
    Tropical fruit,Fruit,Includes,Himsagar,Location
    Mango,Fruit,Coreference,Tropical fruit,Fruit
    Mango variety,Fruit,Includes,Himsagar,Location
    Mango,Fruit,Coreference,Mango variety,Fruit
    Mango fruit,Fruit,Includes,Chausa mango,Fruit
    Mango,Fruit,Coreference,Mango fruit,Fruit
    Mango,Fruit,Includes,Chausa mango,Fruit
    Mango,Fruit,Includes,Chausa mango,Fruit
    Mango,Fruit,Includes,Chausa mango,Fruit
    Mango,Fruit,Includes,Chausa mango,Fruit
    Fruit,O,Includes,Langda,Location
    Mango,Fruit,Coreference,Fruit,O
    Langra,Location,Coreference,Langda,Location
    Mango,Fruit,Includes,Langra variety,Crop
    Fruit,O,Includes,Langra type,Crop
    Mangoes,Crop,Includes,Langra cultivar,Crop
    Fruit,O,Includes,Langra mango,Crop
    Mango,Fruit,Origin_Of,India,Location
    Alphonso,Location,Coreference,Mango,Fruit
    King of Fruits,O,Origin_Of,India,Location
    Alphonso,Location,Coreference,King of Fruits,O
    Tropical fruit,Fruit,Origin_Of,India,Location
    Alphonso,Location,Coreference,Tropical fruit,Fruit
    Mangoes,Crop,Origin_Of,India,Location
    Alphonso,Location,Coreference,Mangoes,Crop
    Mango variety,Fruit,Origin_Of,India,Location
    Alphonso,Location,Coreference,Mango variety,Fruit
    Fruit,O,Seasonal,summer,Season
    Guava,Location,Coreference,Fruit,O
    Guava fruit,Fruit,Seasonal,summer,Season
    Guava,Location,Coreference,Guava fruit,Fruit
    Fruit,O,Seasonal,warm season,Season
    Guava,Location,Coreference,Fruit,O
    summer,Season,Coreference,warm season,Season
    Guava,Location,Seasonal,hot season,Season
    summer,Season,Coreference,hot season,Season
    Fruit,O,Seasonal,summertime,Season
    Guava,Location,Coreference,Fruit,O
    summer,Season,Coreference,summertime,Season
    date palms,Crop,Origin_Of,Middle East,Location
    dates,O,Coreference,date palms,Crop
    date fruit,Fruit,Origin_Of,Middle East,Location
    dates,O,Coreference,date fruit,Fruit
    Fruit,O,Seasonal,autumn,Season
    oranges,Fruit,Coreference,Fruit,O
    Oranges,Fruit,Seasonal,fall,Season
    oranges,Fruit,Coreference,Oranges,Fruit
    autumn,Season,Coreference,fall,Season
    Citrus fruits,Fruit,Seasonal,autumn,Season
    oranges,Fruit,Coreference,Citrus fruits,Fruit
    Tangerines,Crop,Seasonal,autumn,Season
    oranges,Fruit,Coreference,Tangerines,Crop
    pomegranate,Fruit,Seasonal,winter,Season
    winter,Season,Coreference,winter,Season
    pomegranate fruit,Fruit,Seasonal,winter,Season
    pomegranate,Fruit,Coreference,pomegranate fruit,Fruit
    pomegranate,Fruit,Seasonal,winter,Season
    apple,Fruit,Origin_Of,Kashmir valley,Location
    Kashmir region,Location,Coreference,Kashmir valley,Location
    apple,Fruit,Origin_Of,Kashmir territory,Location
    Kashmir region,Location,Coreference,Kashmir territory,Location
    apple,Fruit,Origin_Of,Kashmir area,Location
    Kashmir region,Location,Coreference,Kashmir area,Location
    apple,Fruit,Origin_Of,Kashmir district,Location
    Kashmir region,Location,Coreference,Kashmir district,Location
    apple,Fruit,Origin_Of,Kashmir locality,Location
    Kashmir region,Location,Coreference,Kashmir locality,Location
    apple,Fruit,Seasonal,winter,Season
    winter,Season,Coreference,winter,Season
    Apple,O,Seasonal,cold season,Season
    apple,Fruit,Coreference,Apple,O
    winter,Season,Coreference,cold season,Season
    Fruit,O,Seasonal,winter,Season
    apple,Fruit,Coreference,Fruit,O
    Apple,O,Seasonal,chilly season,Season
    apple,Fruit,Coreference,Apple,O
    winter,Season,Coreference,chilly season,Season
    Fruit of Fig tree,O,Seasonal,summers,Season
    Fig,O,Coreference,Fruit of Fig tree,O
    Fig,O,Seasonal,summer,Season
    Fruiting Fig tree,Organism,Seasonal,summers,Season
    Fig,O,Coreference,Fruiting Fig tree,Organism
    Fig,O,Seasonal,hot season,Season
    sweet,O,Includes,cakes,O
    dessert,O,Coreference,sweet,O
    treat,O,Includes,cakes,O
    dessert,O,Coreference,treat,O
    pudding,O,Includes,cakes,O
    dessert,O,Coreference,pudding,O
    confection,Food_Item,Includes,cakes,O
    dessert,O,Coreference,confection,Food_Item
    dessert,O,Includes,pastries,Food_Item
    Sweet,O,Includes,cupcakes,Food_Item
    dessert,O,Coreference,Sweet,O
    Dessert,O,Includes,mini cakes,Food_Item
    dessert,O,Coreference,Dessert,O
    cupcakes,Food_Item,Coreference,mini cakes,Food_Item
    Food_Item,O,Includes,cupcakes,Food_Item
    dessert,O,Coreference,Food_Item,O
    Treat,O,Includes,cupcakes,Food_Item
    dessert,O,Coreference,Treat,O
    Dessert,O,Includes,frosted pastries,Food_Item
    dessert,O,Coreference,Dessert,O
    cupcakes,Food_Item,Coreference,frosted pastries,Food_Item
    sweet,O,Includes,muffins,Food_Item
    dessert,O,Coreference,sweet,O
    treat,O,Includes,muffins,Food_Item
    dessert,O,Coreference,treat,O
    pastry,O,Includes,muffins,Food_Item
    dessert,O,Coreference,pastry,O
    delicacy,Food_Item,Includes,muffins,Food_Item
    dessert,O,Coreference,delicacy,Food_Item
    dessert,O,Includes,cupcakes,Food_Item
    sweet,O,Includes,cherry toppings,Food_Item
    dessert,O,Coreference,sweet,O
    cherry sauces,Food_Item,Coreference,cherry toppings,Food_Item
    treat,O,Includes,cherry sauces,Food_Item
    dessert,O,Coreference,treat,O
    dessert,O,Includes,cherry syrups,Food_Item
    confection,Food_Item,Includes,cherry sauces,Food_Item
    pastry,O,Includes,cherry toppings,Food_Item
    Fruit of Jackfruit tree,O,Seasonal,spring,Season
    Jackfruit,Fruit,Coreference,Fruit of Jackfruit tree,O
    Jackfruit,Fruit,Seasonal,springtime,Season
    spring,Season,Coreference,springtime,Season
    Jackfruit,Fruit,Seasonal,spring season,Season
    spring,Season,Coreference,spring season,Season
    Jackfruit,Fruit,Seasonal,spring period,Season
    spring,Season,Coreference,spring period,Season
    Jackfruit,Fruit,Seasonal,springtime period,Season
    spring,Season,Coreference,springtime period,Season
    Jackfruit,Fruit,Seasonal,summer,Season
    Jackfruit,Fruit,Origin_Of,India,Location
    Apricots,Fruit,Seasonal,springtime,Season
    apricots,Fruit,Coreference,Apricots,Fruit
    spring,Season,Coreference,springtime,Season
    Apricot fruits,Fruit,Seasonal,spring,Season
    apricots,Fruit,Coreference,Apricot fruits,Fruit
    Apricot harvest,Season,Seasonal,spring,Season
    apricots,Fruit,Coreference,Apricot harvest,Season
    Apricot crop,Crop,Seasonal,spring,Season
    apricots,Fruit,Coreference,Apricot crop,Crop
    Apricot,Crop,Seasonal,spring season,Season
    apricots,Fruit,Coreference,Apricot,Crop
    spring,Season,Coreference,spring season,Season
    oil palm tree,Crop,Origin_Of,tropical rainforest,Location
    oil palm,Crop,Coreference,oil palm tree,Crop
    tropical forest,Location,Coreference,tropical rainforest,Location
    palm oil,Crop,Origin_Of,rainforest,Location
    oil palm,Crop,Coreference,palm oil,Crop
    tropical forest,Location,Coreference,rainforest,Location
    oil-bearing palm,Crop,Origin_Of,tropic forest,Location
    oil palm,Crop,Coreference,oil-bearing palm,Crop
    tropical forest,Location,Coreference,tropic forest,Location
    palm fruit,Crop,Origin_Of,jungle,Location
    oil palm,Crop,Coreference,palm fruit,Crop
    tropical forest,Location,Coreference,jungle,Location
    Elaeis guineensis,Organism,Origin_Of,equatorial forest,Location
    oil palm,Crop,Coreference,Elaeis guineensis,Organism
    tropical forest,Location,Coreference,equatorial forest,Location
    wintertime,Season,Seasonal,winter,Season
    rabi,O,Coreference,wintertime,Season
    crop-growing,Agri_Process,Seasonal,winter,Season
    rabi,O,Coreference,crop-growing,Agri_Process
    hibernal,Season,Seasonal,winter,Season
    rabi,O,Coreference,hibernal,Season
    cold season,Season,Seasonal,winter,Season
    rabi,O,Coreference,cold season,Season
    wintry,Weather,Seasonal,winter,Season
    rabi,O,Coreference,wintry,Weather
    Indian Council of Agricultural Research,Organization,Synonym_Of,Indian Agricultural Research Institute,Organization
    ICAR,Organization,Coreference,Indian Council of Agricultural Research,Organization
    Agricultural Research Council of India,Organization,Synonym_Of,Indian Agricultural Research Institute,Organization
    ICAR,Organization,Coreference,Agricultural Research Council of India,Organization
    Indian Agricultural Research Institute,Organization,Synonym_Of,Indian Council of Agricultural Research,Organization
    ICAR,Organization,Coreference,Indian Agricultural Research Institute,Organization
    Indian Agricultural Research Institute,Organization,Coreference,Indian Council of Agricultural Research,Organization
    National Agricultural Research Institute,Organization,Synonym_Of,Indian Agricultural Research Institute,Organization
    ICAR,Organization,Coreference,National Agricultural Research Institute,Organization
    Indian Farming Research Institute,Organization,Origin_Of,New Delhi,Location
    Indian Agricultural Research Institute,Organization,Coreference,Indian Farming Research Institute,Organization
    Indian Agro Research Institute,Organization,Origin_Of,New Delhi,Location
    Indian Agricultural Research Institute,Organization,Coreference,Indian Agro Research Institute,Organization
    Indian Agriculture Study Institute,Organization,Origin_Of,New Delhi,Location
    Indian Agricultural Research Institute,Organization,Coreference,Indian Agriculture Study Institute,Organization
    Institute of Indian Agriculture Research,Organization,Origin_Of,New Delhi,Location
    Indian Agricultural Research Institute,Organization,Coreference,Institute of Indian Agriculture Research,Organization
    Indian Agricultural Experiment Station,Organization,Origin_Of,New Delhi,Location
    Indian Agricultural Research Institute,Organization,Coreference,Indian Agricultural Experiment Station,Organization
    Fire blight,Disease,Caused_By,Bacterium Erwinia amylovora,Organism
    Fire bligh,Disease,Coreference,Fire blight,Disease
    Erwinia amylovora,Crop,Coreference,Bacterium Erwinia amylovora,Organism
    Fire blight,Disease,Caused_By,Pathogen Erwinia amylovora,O
    Fire bligh,Disease,Coreference,Fire blight,Disease
    Erwinia amylovora,Crop,Coreference,Pathogen Erwinia amylovora,O
    Fire blight,Disease,Caused_By,Microorganism Erwinia amylovora,Organism
    Fire bligh,Disease,Coreference,Fire blight,Disease
    Erwinia amylovora,Crop,Coreference,Microorganism Erwinia amylovora,Organism
    Fire blight,Disease,Caused_By,Bacterial infection caused by Erwinia amylovora,O
    Fire bligh,Disease,Coreference,Fire blight,Disease
    Erwinia amylovora,Crop,Coreference,Bacterial infection caused by Erwinia amylovora,O
    Fire blight,Disease,Caused_By,Erwinia amylovora bacteria,Organism
    Fire bligh,Disease,Coreference,Fire blight,Disease
    Erwinia amylovora,Crop,Coreference,Erwinia amylovora bacteria,Organism
    apple fruit,Fruit,Origin_Of,North America,Location
    apple,Fruit,Coreference,apple fruit,Fruit
    orchard apple,Fruit,Origin_Of,North America,Location
    apple,Fruit,Coreference,orchard apple,Fruit
    pomaceous fruit,Fruit,Origin_Of,North America,Location
    apple,Fruit,Coreference,pomaceous fruit,Fruit
    crab apple,Fruit,Origin_Of,North America,Location
    apple,Fruit,Coreference,crab apple,Fruit
    tree fruit,Crop,Origin_Of,North America,Location
    apple,Fruit,Coreference,tree fruit,Crop
    apple,Fruit,Origin_Of,European continent,Location
    Europe,Location,Coreference,European continent,Location
    apple,Fruit,Origin_Of,European region,Location
    Europe,Location,Coreference,European region,Location
    apple,Fruit,Origin_Of,European land,Location
    Europe,Location,Coreference,European land,Location
    apple,Fruit,Origin_Of,Europe,Location
    Europe,Location,Coreference,Europe,Location
    apple,Fruit,Origin_Of,European soil,Location
    Europe,Location,Coreference,European soil,Location
    apple,Fruit,Origin_Of,NZ,Location
    New Zealand,Location,Coreference,NZ,Location
    fruit,O,Origin_Of,New Zealand,Location
    apple,Fruit,Coreference,fruit,O
    apple,Fruit,Origin_Of,New Zealand,Location
    apple variety,Fruit,Origin_Of,New Zealand,Location
    apple,Fruit,Coreference,apple variety,Fruit
    Fruit,O,Origin_Of,Japan,Location
    apple,Fruit,Coreference,Fruit,O
    Fire blight disease,Disease,Caused_By,warm humid climate,O
    Fire blight,Disease,Coreference,Fire blight disease,Disease
    warm moist weather,O,Coreference,warm humid climate,O
    Fire blight,Disease,Caused_By,hot damp conditions,O
    warm moist weather,O,Coreference,hot damp conditions,O
    Fire blight illness,Disease,Caused_By,warm moist weather,O
    Fire blight,Disease,Coreference,Fire blight illness,Disease
    Fire blight infection,Disease,Caused_By,warm humid environment,O
    Fire blight,Disease,Coreference,Fire blight infection,Disease
    warm moist weather,O,Coreference,warm humid environment,O
    Fire blight,Disease,Caused_By,hot moist weather,O
    warm moist weather,O,Coreference,hot moist weather,O
    Antibiotic,O,Helps_In,Fire blight,Disease
    Streptomycin,Chemical,Coreference,Antibiotic,O
    Antimicrobial,Chemical,Helps_In,Fire blight,Disease
    Streptomycin,Chemical,Coreference,Antimicrobial,Chemical
    Strep,Disease,Helps_In,Fire blight,Disease
    Streptomycin,Chemical,Coreference,Strep,Disease
    Drug,O,Helps_In,Fire blight,Disease
    Streptomycin,Chemical,Coreference,Drug,O
    Medication,O,Helps_In,Fire blight,Disease
    Streptomycin,Chemical,Coreference,Medication,O
    Moist climates,O,Helps_In,bacterial blight,Disease
    humid environments,Weather,Coreference,Moist climates,O
    Damp conditions,O,Helps_In,bacterial blight,Disease
    humid environments,Weather,Coreference,Damp conditions,O
    Humid climate,O,Helps_In,bacterial blight,Disease
    humid environments,Weather,Coreference,Humid climate,O
    Wet environments,Location,Helps_In,bacterial blight,Disease
    humid environments,Weather,Coreference,Wet environments,Location
    Soggy weather,O,Helps_In,bacterial blight,Disease
    humid environments,Weather,Coreference,Soggy weather,O
    bacterial infection,O,Origin_Of,Asia,Location
    bacterial blight,Disease,Coreference,bacterial infection,O
    bacterial disease,Disease,Origin_Of,Asia,Location
    bacterial blight,Disease,Coreference,bacterial disease,Disease
    bacterial wilt,Disease,Origin_Of,Asia,Location
    bacterial blight,Disease,Coreference,bacterial wilt,Disease
    bacterial outbreak,O,Origin_Of,Asia,Location
    bacterial blight,Disease,Coreference,bacterial outbreak,O
    bacterial epidemic,O,Origin_Of,Asia,Location
    bacterial blight,Disease,Coreference,bacterial epidemic,O
    Gardening,O,Conjunction,Farming,O
    Horticultural,O,Conjunction,Agricultural,O
    Orcharding,Agri_Process,Conjunction,Agricultural,O
    Floricultural,O,Conjunction,Agricultural,O
    Gardening,O,Conjunction,Farming,O
    horticultural,Agri_Process,Conjunction,Agricultural,O
    Gardening,O,Conjunction,Farming,O
    Horticultural,O,Conjunction,Agricultural,O
    Orcharding,Agri_Process,Conjunction,Agricultural,O
    Floricultural,O,Conjunction,Agricultural,O
    Gardening,O,Conjunction,Farming,O
    horticultural,Agri_Process,Conjunction,Agricultural,O
    Soil pollution,Agri_Pollution,Caused_By,horticultural practices,Agri_Process
    Contaminating farmlands,O,Caused_By,horticulture,Agri_Process
    Soil contamination,Agri_Pollution,Caused_By,horticultural activities,Agri_Process
    Polluting the earth,O,Caused_By,gardening,O
    Soil degradation,Agri_Pollution,Caused_By,horticultural methods,Agri_Process
    contaminating the soils,Agri_Pollution,Caused_By,horticultural,Agri_Process
    Gardening,O,Conjunction,Farming,O
    Horticultural,O,Conjunction,Agricultural,O
    Orcharding,Agri_Process,Conjunction,Agricultural,O
    Floricultural,O,Conjunction,Agricultural,O
    Gardening,O,Conjunction,Farming,O
    horticultural,Agri_Process,Conjunction,Agricultural,O
    Soil pollution,Agri_Pollution,Caused_By,horticultural practices,Agri_Process
    Contaminating farmlands,O,Caused_By,horticulture,Agri_Process
    Soil contamination,Agri_Pollution,Caused_By,horticultural activities,Agri_Process
    Polluting the earth,O,Caused_By,gardening,O
    Soil degradation,Agri_Pollution,Caused_By,horticultural methods,Agri_Process
    contaminating the soils,Agri_Pollution,Caused_By,horticultural,Agri_Process
    soil erosion,O,Caused_By,horticultural practices,Agri_Process
    deposits,O,Caused_By,horticultural activities,Agri_Process
    silt,O,Caused_By,horticultural impact,Agri_Process
    particle sediments,O,Caused_By,horticultural methods,Agri_Process
    sediments,O,Caused_By,horticultural,Agri_Process
    Gardening,O,Conjunction,Farming,O
    Horticultural,O,Conjunction,Agricultural,O
    Orcharding,Agri_Process,Conjunction,Agricultural,O
    Floricultural,O,Conjunction,Agricultural,O
    Gardening,O,Conjunction,Farming,O
    horticultural,Agri_Process,Conjunction,Agricultural,O
    Soil pollution,Agri_Pollution,Caused_By,horticultural practices,Agri_Process
    Contaminating farmlands,O,Caused_By,horticulture,Agri_Process
    Soil contamination,Agri_Pollution,Caused_By,horticultural activities,Agri_Process
    Polluting the earth,O,Caused_By,gardening,O
    Soil degradation,Agri_Pollution,Caused_By,horticultural methods,Agri_Process
    contaminating the soils,Agri_Pollution,Caused_By,horticultural,Agri_Process
    soil erosion,O,Caused_By,horticultural practices,Agri_Process
    deposits,O,Caused_By,horticultural activities,Agri_Process
    silt,O,Caused_By,horticultural impact,Agri_Process
    particle sediments,O,Caused_By,horticultural methods,Agri_Process
    sediments,O,Caused_By,horticultural,Agri_Process
    Polluting the land,O,Caused_By,Agricultural,O
    Soil contamination,Agri_Pollution,Caused_By,Agricultural,O
    Contaminating the farmland,O,Caused_By,Agricultural,O
    Land pollution,O,Caused_By,Agricultural,O
    Soil degradation,Agri_Pollution,Caused_By,Agricultural,O
    contaminating the soils,Agri_Pollution,Caused_By,Agricultural,O
    Gardening,O,Conjunction,Farming,O
    Horticultural,O,Conjunction,Agricultural,O
    Orcharding,Agri_Process,Conjunction,Agricultural,O
    Floricultural,O,Conjunction,Agricultural,O
    Gardening,O,Conjunction,Farming,O
    horticultural,Agri_Process,Conjunction,Agricultural,O
    Soil pollution,Agri_Pollution,Caused_By,horticultural practices,Agri_Process
    Contaminating farmlands,O,Caused_By,horticulture,Agri_Process
    Soil contamination,Agri_Pollution,Caused_By,horticultural activities,Agri_Process
    Polluting the earth,O,Caused_By,gardening,O
    Soil degradation,Agri_Pollution,Caused_By,horticultural methods,Agri_Process
    contaminating the soils,Agri_Pollution,Caused_By,horticultural,Agri_Process
    soil erosion,O,Caused_By,horticultural practices,Agri_Process
    deposits,O,Caused_By,horticultural activities,Agri_Process
    silt,O,Caused_By,horticultural impact,Agri_Process
    particle sediments,O,Caused_By,horticultural methods,Agri_Process
    sediments,O,Caused_By,horticultural,Agri_Process
    Polluting the land,O,Caused_By,Agricultural,O
    Soil contamination,Agri_Pollution,Caused_By,Agricultural,O
    Contaminating the farmland,O,Caused_By,Agricultural,O
    Land pollution,O,Caused_By,Agricultural,O
    Soil degradation,Agri_Pollution,Caused_By,Agricultural,O
    contaminating the soils,Agri_Pollution,Caused_By,Agricultural,O
    sedimentation,O,Caused_By,Farming practices,O
    deposits,O,Caused_By,Agricultural activities,O
    silt,O,Caused_By,Agriculture,Agri_Process
    alluvium,Crop,Caused_By,Farming,O
    settled particles,O,Caused_By,Agricultural practices,O
    sediments,O,Caused_By,Agricultural,O
    Gardening,O,Conjunction,Farming,O
    Horticultural,O,Conjunction,Agricultural,O
    Orcharding,Agri_Process,Conjunction,Agricultural,O
    Floricultural,O,Conjunction,Agricultural,O
    Gardening,O,Conjunction,Farming,O
    horticultural,Agri_Process,Conjunction,Agricultural,O
    Soil pollution,Agri_Pollution,Caused_By,horticultural practices,Agri_Process
    Contaminating farmlands,O,Caused_By,horticulture,Agri_Process
    Soil contamination,Agri_Pollution,Caused_By,horticultural activities,Agri_Process
    Polluting the earth,O,Caused_By,gardening,O
    Soil degradation,Agri_Pollution,Caused_By,horticultural methods,Agri_Process
    contaminating the soils,Agri_Pollution,Caused_By,horticultural,Agri_Process
    soil erosion,O,Caused_By,horticultural practices,Agri_Process
    deposits,O,Caused_By,horticultural activities,Agri_Process
    silt,O,Caused_By,horticultural impact,Agri_Process
    particle sediments,O,Caused_By,horticultural methods,Agri_Process
    sediments,O,Caused_By,horticultural,Agri_Process
    Polluting the land,O,Caused_By,Agricultural,O
    Soil contamination,Agri_Pollution,Caused_By,Agricultural,O
    Contaminating the farmland,O,Caused_By,Agricultural,O
    Land pollution,O,Caused_By,Agricultural,O
    Soil degradation,Agri_Pollution,Caused_By,Agricultural,O
    contaminating the soils,Agri_Pollution,Caused_By,Agricultural,O
    sedimentation,O,Caused_By,Farming practices,O
    deposits,O,Caused_By,Agricultural activities,O
    silt,O,Caused_By,Agriculture,Agri_Process
    alluvium,Crop,Caused_By,Farming,O
    settled particles,O,Caused_By,Agricultural practices,O
    sediments,O,Caused_By,Agricultural,O
    Agri_Pollution,Agri_Process,Includes,natural and synthetic fertilizers,O
    Agricultural pollutants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollution,Agri_Process,Includes,natural and chemical fertilizers,O
    Agricultural contaminants,Agri_Pollution,Includes,natural and chemical fertilizers,O
    Agri_Pollution,Agri_Process,Includes,organic and chemical fertilizers,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Gardening,O,Conjunction,Farming,O
    Horticultural,O,Conjunction,Agricultural,O
    Orcharding,Agri_Process,Conjunction,Agricultural,O
    Floricultural,O,Conjunction,Agricultural,O
    Gardening,O,Conjunction,Farming,O
    horticultural,Agri_Process,Conjunction,Agricultural,O
    Soil pollution,Agri_Pollution,Caused_By,horticultural practices,Agri_Process
    Contaminating farmlands,O,Caused_By,horticulture,Agri_Process
    Soil contamination,Agri_Pollution,Caused_By,horticultural activities,Agri_Process
    Polluting the earth,O,Caused_By,gardening,O
    Soil degradation,Agri_Pollution,Caused_By,horticultural methods,Agri_Process
    contaminating the soils,Agri_Pollution,Caused_By,horticultural,Agri_Process
    soil erosion,O,Caused_By,horticultural practices,Agri_Process
    deposits,O,Caused_By,horticultural activities,Agri_Process
    silt,O,Caused_By,horticultural impact,Agri_Process
    particle sediments,O,Caused_By,horticultural methods,Agri_Process
    sediments,O,Caused_By,horticultural,Agri_Process
    Polluting the land,O,Caused_By,Agricultural,O
    Soil contamination,Agri_Pollution,Caused_By,Agricultural,O
    Contaminating the farmland,O,Caused_By,Agricultural,O
    Land pollution,O,Caused_By,Agricultural,O
    Soil degradation,Agri_Pollution,Caused_By,Agricultural,O
    contaminating the soils,Agri_Pollution,Caused_By,Agricultural,O
    sedimentation,O,Caused_By,Farming practices,O
    deposits,O,Caused_By,Agricultural activities,O
    silt,O,Caused_By,Agriculture,Agri_Process
    alluvium,Crop,Caused_By,Farming,O
    settled particles,O,Caused_By,Agricultural practices,O
    sediments,O,Caused_By,Agricultural,O
    Agri_Pollution,Agri_Process,Includes,natural and synthetic fertilizers,O
    Agricultural pollutants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollution,Agri_Process,Includes,natural and chemical fertilizers,O
    Agricultural contaminants,Agri_Pollution,Includes,natural and chemical fertilizers,O
    Agri_Pollution,Agri_Process,Includes,organic and chemical fertilizers,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollutants,Crop,Includes,insecticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,pest control substances,Agri_Pollution
    Agricultural pollution,O,Includes,herbicides,Chemical
    Agri_Pollutants,Crop,Includes,chemical pesticides,Chemical
    Farming contaminants,Agri_Pollution,Includes,pest management products,O
    Agricultural contaminants,Agri_Pollution,Includes,pesticides,Chemical
    Gardening,O,Conjunction,Farming,O
    Horticultural,O,Conjunction,Agricultural,O
    Orcharding,Agri_Process,Conjunction,Agricultural,O
    Floricultural,O,Conjunction,Agricultural,O
    Gardening,O,Conjunction,Farming,O
    horticultural,Agri_Process,Conjunction,Agricultural,O
    Soil pollution,Agri_Pollution,Caused_By,horticultural practices,Agri_Process
    Contaminating farmlands,O,Caused_By,horticulture,Agri_Process
    Soil contamination,Agri_Pollution,Caused_By,horticultural activities,Agri_Process
    Polluting the earth,O,Caused_By,gardening,O
    Soil degradation,Agri_Pollution,Caused_By,horticultural methods,Agri_Process
    contaminating the soils,Agri_Pollution,Caused_By,horticultural,Agri_Process
    soil erosion,O,Caused_By,horticultural practices,Agri_Process
    deposits,O,Caused_By,horticultural activities,Agri_Process
    silt,O,Caused_By,horticultural impact,Agri_Process
    particle sediments,O,Caused_By,horticultural methods,Agri_Process
    sediments,O,Caused_By,horticultural,Agri_Process
    Polluting the land,O,Caused_By,Agricultural,O
    Soil contamination,Agri_Pollution,Caused_By,Agricultural,O
    Contaminating the farmland,O,Caused_By,Agricultural,O
    Land pollution,O,Caused_By,Agricultural,O
    Soil degradation,Agri_Pollution,Caused_By,Agricultural,O
    contaminating the soils,Agri_Pollution,Caused_By,Agricultural,O
    sedimentation,O,Caused_By,Farming practices,O
    deposits,O,Caused_By,Agricultural activities,O
    silt,O,Caused_By,Agriculture,Agri_Process
    alluvium,Crop,Caused_By,Farming,O
    settled particles,O,Caused_By,Agricultural practices,O
    sediments,O,Caused_By,Agricultural,O
    Agri_Pollution,Agri_Process,Includes,natural and synthetic fertilizers,O
    Agricultural pollutants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollution,Agri_Process,Includes,natural and chemical fertilizers,O
    Agricultural contaminants,Agri_Pollution,Includes,natural and chemical fertilizers,O
    Agri_Pollution,Agri_Process,Includes,organic and chemical fertilizers,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollutants,Crop,Includes,insecticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,pest control substances,Agri_Pollution
    Agricultural pollution,O,Includes,herbicides,Chemical
    Agri_Pollutants,Crop,Includes,chemical pesticides,Chemical
    Farming contaminants,Agri_Pollution,Includes,pest management products,O
    Agricultural contaminants,Agri_Pollution,Includes,pesticides,Chemical
    Agri_Pollution,Agri_Process,Includes,weed killers,O
    Agricultural pollutants,Agri_Pollution,Includes,herbicides,Chemical
    Farm pollution,O,Includes,weed control chemicals,Chemical
    Agri_Contaminants,Chemical,Includes,herbicides,Chemical
    Agricultural pollution,O,Includes,weed management agents,Person
    Agricultural contaminants,Agri_Pollution,Includes,herbicides,Chemical
    Gardening,O,Conjunction,Farming,O
    Horticultural,O,Conjunction,Agricultural,O
    Orcharding,Agri_Process,Conjunction,Agricultural,O
    Floricultural,O,Conjunction,Agricultural,O
    Gardening,O,Conjunction,Farming,O
    horticultural,Agri_Process,Conjunction,Agricultural,O
    Soil pollution,Agri_Pollution,Caused_By,horticultural practices,Agri_Process
    Contaminating farmlands,O,Caused_By,horticulture,Agri_Process
    Soil contamination,Agri_Pollution,Caused_By,horticultural activities,Agri_Process
    Polluting the earth,O,Caused_By,gardening,O
    Soil degradation,Agri_Pollution,Caused_By,horticultural methods,Agri_Process
    contaminating the soils,Agri_Pollution,Caused_By,horticultural,Agri_Process
    soil erosion,O,Caused_By,horticultural practices,Agri_Process
    deposits,O,Caused_By,horticultural activities,Agri_Process
    silt,O,Caused_By,horticultural impact,Agri_Process
    particle sediments,O,Caused_By,horticultural methods,Agri_Process
    sediments,O,Caused_By,horticultural,Agri_Process
    Polluting the land,O,Caused_By,Agricultural,O
    Soil contamination,Agri_Pollution,Caused_By,Agricultural,O
    Contaminating the farmland,O,Caused_By,Agricultural,O
    Land pollution,O,Caused_By,Agricultural,O
    Soil degradation,Agri_Pollution,Caused_By,Agricultural,O
    contaminating the soils,Agri_Pollution,Caused_By,Agricultural,O
    sedimentation,O,Caused_By,Farming practices,O
    deposits,O,Caused_By,Agricultural activities,O
    silt,O,Caused_By,Agriculture,Agri_Process
    alluvium,Crop,Caused_By,Farming,O
    settled particles,O,Caused_By,Agricultural practices,O
    sediments,O,Caused_By,Agricultural,O
    Agri_Pollution,Agri_Process,Includes,natural and synthetic fertilizers,O
    Agricultural pollutants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollution,Agri_Process,Includes,natural and chemical fertilizers,O
    Agricultural contaminants,Agri_Pollution,Includes,natural and chemical fertilizers,O
    Agri_Pollution,Agri_Process,Includes,organic and chemical fertilizers,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollutants,Crop,Includes,insecticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,pest control substances,Agri_Pollution
    Agricultural pollution,O,Includes,herbicides,Chemical
    Agri_Pollutants,Crop,Includes,chemical pesticides,Chemical
    Farming contaminants,Agri_Pollution,Includes,pest management products,O
    Agricultural contaminants,Agri_Pollution,Includes,pesticides,Chemical
    Agri_Pollution,Agri_Process,Includes,weed killers,O
    Agricultural pollutants,Agri_Pollution,Includes,herbicides,Chemical
    Farm pollution,O,Includes,weed control chemicals,Chemical
    Agri_Contaminants,Chemical,Includes,herbicides,Chemical
    Agricultural pollution,O,Includes,weed management agents,Person
    Agricultural contaminants,Agri_Pollution,Includes,herbicides,Chemical
    Agri_Pollution,Agri_Process,Includes,pesticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,insect repellents,Chemical
    Agri_Pollution,Agri_Process,Includes,bug killers,O
    Agricultural contaminants,Agri_Pollution,Includes,pest management chemicals,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,insecticides,Chemical
    Gardening,O,Conjunction,Farming,O
    Horticultural,O,Conjunction,Agricultural,O
    Orcharding,Agri_Process,Conjunction,Agricultural,O
    Floricultural,O,Conjunction,Agricultural,O
    Gardening,O,Conjunction,Farming,O
    horticultural,Agri_Process,Conjunction,Agricultural,O
    Soil pollution,Agri_Pollution,Caused_By,horticultural practices,Agri_Process
    Contaminating farmlands,O,Caused_By,horticulture,Agri_Process
    Soil contamination,Agri_Pollution,Caused_By,horticultural activities,Agri_Process
    Polluting the earth,O,Caused_By,gardening,O
    Soil degradation,Agri_Pollution,Caused_By,horticultural methods,Agri_Process
    contaminating the soils,Agri_Pollution,Caused_By,horticultural,Agri_Process
    soil erosion,O,Caused_By,horticultural practices,Agri_Process
    deposits,O,Caused_By,horticultural activities,Agri_Process
    silt,O,Caused_By,horticultural impact,Agri_Process
    particle sediments,O,Caused_By,horticultural methods,Agri_Process
    sediments,O,Caused_By,horticultural,Agri_Process
    Polluting the land,O,Caused_By,Agricultural,O
    Soil contamination,Agri_Pollution,Caused_By,Agricultural,O
    Contaminating the farmland,O,Caused_By,Agricultural,O
    Land pollution,O,Caused_By,Agricultural,O
    Soil degradation,Agri_Pollution,Caused_By,Agricultural,O
    contaminating the soils,Agri_Pollution,Caused_By,Agricultural,O
    sedimentation,O,Caused_By,Farming practices,O
    deposits,O,Caused_By,Agricultural activities,O
    silt,O,Caused_By,Agriculture,Agri_Process
    alluvium,Crop,Caused_By,Farming,O
    settled particles,O,Caused_By,Agricultural practices,O
    sediments,O,Caused_By,Agricultural,O
    Agri_Pollution,Agri_Process,Includes,natural and synthetic fertilizers,O
    Agricultural pollutants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollution,Agri_Process,Includes,natural and chemical fertilizers,O
    Agricultural contaminants,Agri_Pollution,Includes,natural and chemical fertilizers,O
    Agri_Pollution,Agri_Process,Includes,organic and chemical fertilizers,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollutants,Crop,Includes,insecticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,pest control substances,Agri_Pollution
    Agricultural pollution,O,Includes,herbicides,Chemical
    Agri_Pollutants,Crop,Includes,chemical pesticides,Chemical
    Farming contaminants,Agri_Pollution,Includes,pest management products,O
    Agricultural contaminants,Agri_Pollution,Includes,pesticides,Chemical
    Agri_Pollution,Agri_Process,Includes,weed killers,O
    Agricultural pollutants,Agri_Pollution,Includes,herbicides,Chemical
    Farm pollution,O,Includes,weed control chemicals,Chemical
    Agri_Contaminants,Chemical,Includes,herbicides,Chemical
    Agricultural pollution,O,Includes,weed management agents,Person
    Agricultural contaminants,Agri_Pollution,Includes,herbicides,Chemical
    Agri_Pollution,Agri_Process,Includes,pesticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,insect repellents,Chemical
    Agri_Pollution,Agri_Process,Includes,bug killers,O
    Agricultural contaminants,Agri_Pollution,Includes,pest management chemicals,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,insecticides,Chemical
    farm residue,O,Includes,animal wastes,Agri_Waste
    plant debris,O,Includes,animal wastes,Agri_Waste
    waste material,O,Includes,animal wastes,Agri_Waste
    agricultural waste,O,Includes,animal wastes,Agri_Waste
    biodegradable waste,Agri_Waste,Includes,animal wastes,Agri_Waste
    organic matter,Natural_Resource,Includes,animal wastes,Agri_Waste
    Gardening,O,Conjunction,Farming,O
    Horticultural,O,Conjunction,Agricultural,O
    Orcharding,Agri_Process,Conjunction,Agricultural,O
    Floricultural,O,Conjunction,Agricultural,O
    Gardening,O,Conjunction,Farming,O
    horticultural,Agri_Process,Conjunction,Agricultural,O
    Soil pollution,Agri_Pollution,Caused_By,horticultural practices,Agri_Process
    Contaminating farmlands,O,Caused_By,horticulture,Agri_Process
    Soil contamination,Agri_Pollution,Caused_By,horticultural activities,Agri_Process
    Polluting the earth,O,Caused_By,gardening,O
    Soil degradation,Agri_Pollution,Caused_By,horticultural methods,Agri_Process
    contaminating the soils,Agri_Pollution,Caused_By,horticultural,Agri_Process
    soil erosion,O,Caused_By,horticultural practices,Agri_Process
    deposits,O,Caused_By,horticultural activities,Agri_Process
    silt,O,Caused_By,horticultural impact,Agri_Process
    particle sediments,O,Caused_By,horticultural methods,Agri_Process
    sediments,O,Caused_By,horticultural,Agri_Process
    Polluting the land,O,Caused_By,Agricultural,O
    Soil contamination,Agri_Pollution,Caused_By,Agricultural,O
    Contaminating the farmland,O,Caused_By,Agricultural,O
    Land pollution,O,Caused_By,Agricultural,O
    Soil degradation,Agri_Pollution,Caused_By,Agricultural,O
    contaminating the soils,Agri_Pollution,Caused_By,Agricultural,O
    sedimentation,O,Caused_By,Farming practices,O
    deposits,O,Caused_By,Agricultural activities,O
    silt,O,Caused_By,Agriculture,Agri_Process
    alluvium,Crop,Caused_By,Farming,O
    settled particles,O,Caused_By,Agricultural practices,O
    sediments,O,Caused_By,Agricultural,O
    Agri_Pollution,Agri_Process,Includes,natural and synthetic fertilizers,O
    Agricultural pollutants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollution,Agri_Process,Includes,natural and chemical fertilizers,O
    Agricultural contaminants,Agri_Pollution,Includes,natural and chemical fertilizers,O
    Agri_Pollution,Agri_Process,Includes,organic and chemical fertilizers,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollutants,Crop,Includes,insecticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,pest control substances,Agri_Pollution
    Agricultural pollution,O,Includes,herbicides,Chemical
    Agri_Pollutants,Crop,Includes,chemical pesticides,Chemical
    Farming contaminants,Agri_Pollution,Includes,pest management products,O
    Agricultural contaminants,Agri_Pollution,Includes,pesticides,Chemical
    Agri_Pollution,Agri_Process,Includes,weed killers,O
    Agricultural pollutants,Agri_Pollution,Includes,herbicides,Chemical
    Farm pollution,O,Includes,weed control chemicals,Chemical
    Agri_Contaminants,Chemical,Includes,herbicides,Chemical
    Agricultural pollution,O,Includes,weed management agents,Person
    Agricultural contaminants,Agri_Pollution,Includes,herbicides,Chemical
    Agri_Pollution,Agri_Process,Includes,pesticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,insect repellents,Chemical
    Agri_Pollution,Agri_Process,Includes,bug killers,O
    Agricultural contaminants,Agri_Pollution,Includes,pest management chemicals,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,insecticides,Chemical
    farm residue,O,Includes,animal wastes,Agri_Waste
    plant debris,O,Includes,animal wastes,Agri_Waste
    waste material,O,Includes,animal wastes,Agri_Waste
    agricultural waste,O,Includes,animal wastes,Agri_Waste
    biodegradable waste,Agri_Waste,Includes,animal wastes,Agri_Waste
    organic matter,Natural_Resource,Includes,animal wastes,Agri_Waste
    Agri_Residue,Crop,Includes,decomposing botanical substances,O
    Organic debris,Agri_Waste,Includes,rotting vegetation,O
    Agri_Waste,Agri_Process,Includes,deteriorating flora matter,O
    Organic material,Natural_Resource,Includes,decayed plant elements,O
    Agricultural waste,O,Includes,decomposing plant material,O
    organic matter,Natural_Resource,Includes,decaying plant material,O
    Gardening,O,Conjunction,Farming,O
    Horticultural,O,Conjunction,Agricultural,O
    Orcharding,Agri_Process,Conjunction,Agricultural,O
    Floricultural,O,Conjunction,Agricultural,O
    Gardening,O,Conjunction,Farming,O
    horticultural,Agri_Process,Conjunction,Agricultural,O
    Soil pollution,Agri_Pollution,Caused_By,horticultural practices,Agri_Process
    Contaminating farmlands,O,Caused_By,horticulture,Agri_Process
    Soil contamination,Agri_Pollution,Caused_By,horticultural activities,Agri_Process
    Polluting the earth,O,Caused_By,gardening,O
    Soil degradation,Agri_Pollution,Caused_By,horticultural methods,Agri_Process
    contaminating the soils,Agri_Pollution,Caused_By,horticultural,Agri_Process
    soil erosion,O,Caused_By,horticultural practices,Agri_Process
    deposits,O,Caused_By,horticultural activities,Agri_Process
    silt,O,Caused_By,horticultural impact,Agri_Process
    particle sediments,O,Caused_By,horticultural methods,Agri_Process
    sediments,O,Caused_By,horticultural,Agri_Process
    Polluting the land,O,Caused_By,Agricultural,O
    Soil contamination,Agri_Pollution,Caused_By,Agricultural,O
    Contaminating the farmland,O,Caused_By,Agricultural,O
    Land pollution,O,Caused_By,Agricultural,O
    Soil degradation,Agri_Pollution,Caused_By,Agricultural,O
    contaminating the soils,Agri_Pollution,Caused_By,Agricultural,O
    sedimentation,O,Caused_By,Farming practices,O
    deposits,O,Caused_By,Agricultural activities,O
    silt,O,Caused_By,Agriculture,Agri_Process
    alluvium,Crop,Caused_By,Farming,O
    settled particles,O,Caused_By,Agricultural practices,O
    sediments,O,Caused_By,Agricultural,O
    Agri_Pollution,Agri_Process,Includes,natural and synthetic fertilizers,O
    Agricultural pollutants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollution,Agri_Process,Includes,natural and chemical fertilizers,O
    Agricultural contaminants,Agri_Pollution,Includes,natural and chemical fertilizers,O
    Agri_Pollution,Agri_Process,Includes,organic and chemical fertilizers,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollutants,Crop,Includes,insecticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,pest control substances,Agri_Pollution
    Agricultural pollution,O,Includes,herbicides,Chemical
    Agri_Pollutants,Crop,Includes,chemical pesticides,Chemical
    Farming contaminants,Agri_Pollution,Includes,pest management products,O
    Agricultural contaminants,Agri_Pollution,Includes,pesticides,Chemical
    Agri_Pollution,Agri_Process,Includes,weed killers,O
    Agricultural pollutants,Agri_Pollution,Includes,herbicides,Chemical
    Farm pollution,O,Includes,weed control chemicals,Chemical
    Agri_Contaminants,Chemical,Includes,herbicides,Chemical
    Agricultural pollution,O,Includes,weed management agents,Person
    Agricultural contaminants,Agri_Pollution,Includes,herbicides,Chemical
    Agri_Pollution,Agri_Process,Includes,pesticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,insect repellents,Chemical
    Agri_Pollution,Agri_Process,Includes,bug killers,O
    Agricultural contaminants,Agri_Pollution,Includes,pest management chemicals,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,insecticides,Chemical
    farm residue,O,Includes,animal wastes,Agri_Waste
    plant debris,O,Includes,animal wastes,Agri_Waste
    waste material,O,Includes,animal wastes,Agri_Waste
    agricultural waste,O,Includes,animal wastes,Agri_Waste
    biodegradable waste,Agri_Waste,Includes,animal wastes,Agri_Waste
    organic matter,Natural_Resource,Includes,animal wastes,Agri_Waste
    Agri_Residue,Crop,Includes,decomposing botanical substances,O
    Organic debris,Agri_Waste,Includes,rotting vegetation,O
    Agri_Waste,Agri_Process,Includes,deteriorating flora matter,O
    Organic material,Natural_Resource,Includes,decayed plant elements,O
    Agricultural waste,O,Includes,decomposing plant material,O
    organic matter,Natural_Resource,Includes,decaying plant material,O
    irrigation residues,Agri_Waste,Includes,salts,O
    Gardening,O,Conjunction,Farming,O
    Horticultural,O,Conjunction,Agricultural,O
    Orcharding,Agri_Process,Conjunction,Agricultural,O
    Floricultural,O,Conjunction,Agricultural,O
    Gardening,O,Conjunction,Farming,O
    horticultural,Agri_Process,Conjunction,Agricultural,O
    Soil pollution,Agri_Pollution,Caused_By,horticultural practices,Agri_Process
    Contaminating farmlands,O,Caused_By,horticulture,Agri_Process
    Soil contamination,Agri_Pollution,Caused_By,horticultural activities,Agri_Process
    Polluting the earth,O,Caused_By,gardening,O
    Soil degradation,Agri_Pollution,Caused_By,horticultural methods,Agri_Process
    contaminating the soils,Agri_Pollution,Caused_By,horticultural,Agri_Process
    soil erosion,O,Caused_By,horticultural practices,Agri_Process
    deposits,O,Caused_By,horticultural activities,Agri_Process
    silt,O,Caused_By,horticultural impact,Agri_Process
    particle sediments,O,Caused_By,horticultural methods,Agri_Process
    sediments,O,Caused_By,horticultural,Agri_Process
    Polluting the land,O,Caused_By,Agricultural,O
    Soil contamination,Agri_Pollution,Caused_By,Agricultural,O
    Contaminating the farmland,O,Caused_By,Agricultural,O
    Land pollution,O,Caused_By,Agricultural,O
    Soil degradation,Agri_Pollution,Caused_By,Agricultural,O
    contaminating the soils,Agri_Pollution,Caused_By,Agricultural,O
    sedimentation,O,Caused_By,Farming practices,O
    deposits,O,Caused_By,Agricultural activities,O
    silt,O,Caused_By,Agriculture,Agri_Process
    alluvium,Crop,Caused_By,Farming,O
    settled particles,O,Caused_By,Agricultural practices,O
    sediments,O,Caused_By,Agricultural,O
    Agri_Pollution,Agri_Process,Includes,natural and synthetic fertilizers,O
    Agricultural pollutants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollution,Agri_Process,Includes,natural and chemical fertilizers,O
    Agricultural contaminants,Agri_Pollution,Includes,natural and chemical fertilizers,O
    Agri_Pollution,Agri_Process,Includes,organic and chemical fertilizers,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollutants,Crop,Includes,insecticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,pest control substances,Agri_Pollution
    Agricultural pollution,O,Includes,herbicides,Chemical
    Agri_Pollutants,Crop,Includes,chemical pesticides,Chemical
    Farming contaminants,Agri_Pollution,Includes,pest management products,O
    Agricultural contaminants,Agri_Pollution,Includes,pesticides,Chemical
    Agri_Pollution,Agri_Process,Includes,weed killers,O
    Agricultural pollutants,Agri_Pollution,Includes,herbicides,Chemical
    Farm pollution,O,Includes,weed control chemicals,Chemical
    Agri_Contaminants,Chemical,Includes,herbicides,Chemical
    Agricultural pollution,O,Includes,weed management agents,Person
    Agricultural contaminants,Agri_Pollution,Includes,herbicides,Chemical
    Agri_Pollution,Agri_Process,Includes,pesticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,insect repellents,Chemical
    Agri_Pollution,Agri_Process,Includes,bug killers,O
    Agricultural contaminants,Agri_Pollution,Includes,pest management chemicals,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,insecticides,Chemical
    farm residue,O,Includes,animal wastes,Agri_Waste
    plant debris,O,Includes,animal wastes,Agri_Waste
    waste material,O,Includes,animal wastes,Agri_Waste
    agricultural waste,O,Includes,animal wastes,Agri_Waste
    biodegradable waste,Agri_Waste,Includes,animal wastes,Agri_Waste
    organic matter,Natural_Resource,Includes,animal wastes,Agri_Waste
    Agri_Residue,Crop,Includes,decomposing botanical substances,O
    Organic debris,Agri_Waste,Includes,rotting vegetation,O
    Agri_Waste,Agri_Process,Includes,deteriorating flora matter,O
    Organic material,Natural_Resource,Includes,decayed plant elements,O
    Agricultural waste,O,Includes,decomposing plant material,O
    organic matter,Natural_Resource,Includes,decaying plant material,O
    irrigation residues,Agri_Waste,Includes,salts,O
    irrigation waste,O,Includes,trace elements,O
    irrigation leftovers,O,Includes,trace metals,O
    farming residues,O,Includes,trace elements,O
    agricultural waste,O,Includes,trace metals,O
    irrigation byproducts,Agri_Waste,Includes,trace elements,O
    irrigation residues,Agri_Waste,Includes,trace metals,O
    Gardening,O,Conjunction,Farming,O
    Horticultural,O,Conjunction,Agricultural,O
    Orcharding,Agri_Process,Conjunction,Agricultural,O
    Floricultural,O,Conjunction,Agricultural,O
    Gardening,O,Conjunction,Farming,O
    horticultural,Agri_Process,Conjunction,Agricultural,O
    Soil pollution,Agri_Pollution,Caused_By,horticultural practices,Agri_Process
    Contaminating farmlands,O,Caused_By,horticulture,Agri_Process
    Soil contamination,Agri_Pollution,Caused_By,horticultural activities,Agri_Process
    Polluting the earth,O,Caused_By,gardening,O
    Soil degradation,Agri_Pollution,Caused_By,horticultural methods,Agri_Process
    contaminating the soils,Agri_Pollution,Caused_By,horticultural,Agri_Process
    soil erosion,O,Caused_By,horticultural practices,Agri_Process
    deposits,O,Caused_By,horticultural activities,Agri_Process
    silt,O,Caused_By,horticultural impact,Agri_Process
    particle sediments,O,Caused_By,horticultural methods,Agri_Process
    sediments,O,Caused_By,horticultural,Agri_Process
    Polluting the land,O,Caused_By,Agricultural,O
    Soil contamination,Agri_Pollution,Caused_By,Agricultural,O
    Contaminating the farmland,O,Caused_By,Agricultural,O
    Land pollution,O,Caused_By,Agricultural,O
    Soil degradation,Agri_Pollution,Caused_By,Agricultural,O
    contaminating the soils,Agri_Pollution,Caused_By,Agricultural,O
    sedimentation,O,Caused_By,Farming practices,O
    deposits,O,Caused_By,Agricultural activities,O
    silt,O,Caused_By,Agriculture,Agri_Process
    alluvium,Crop,Caused_By,Farming,O
    settled particles,O,Caused_By,Agricultural practices,O
    sediments,O,Caused_By,Agricultural,O
    Agri_Pollution,Agri_Process,Includes,natural and synthetic fertilizers,O
    Agricultural pollutants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollution,Agri_Process,Includes,natural and chemical fertilizers,O
    Agricultural contaminants,Agri_Pollution,Includes,natural and chemical fertilizers,O
    Agri_Pollution,Agri_Process,Includes,organic and chemical fertilizers,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollutants,Crop,Includes,insecticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,pest control substances,Agri_Pollution
    Agricultural pollution,O,Includes,herbicides,Chemical
    Agri_Pollutants,Crop,Includes,chemical pesticides,Chemical
    Farming contaminants,Agri_Pollution,Includes,pest management products,O
    Agricultural contaminants,Agri_Pollution,Includes,pesticides,Chemical
    Agri_Pollution,Agri_Process,Includes,weed killers,O
    Agricultural pollutants,Agri_Pollution,Includes,herbicides,Chemical
    Farm pollution,O,Includes,weed control chemicals,Chemical
    Agri_Contaminants,Chemical,Includes,herbicides,Chemical
    Agricultural pollution,O,Includes,weed management agents,Person
    Agricultural contaminants,Agri_Pollution,Includes,herbicides,Chemical
    Agri_Pollution,Agri_Process,Includes,pesticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,insect repellents,Chemical
    Agri_Pollution,Agri_Process,Includes,bug killers,O
    Agricultural contaminants,Agri_Pollution,Includes,pest management chemicals,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,insecticides,Chemical
    farm residue,O,Includes,animal wastes,Agri_Waste
    plant debris,O,Includes,animal wastes,Agri_Waste
    waste material,O,Includes,animal wastes,Agri_Waste
    agricultural waste,O,Includes,animal wastes,Agri_Waste
    biodegradable waste,Agri_Waste,Includes,animal wastes,Agri_Waste
    organic matter,Natural_Resource,Includes,animal wastes,Agri_Waste
    Agri_Residue,Crop,Includes,decomposing botanical substances,O
    Organic debris,Agri_Waste,Includes,rotting vegetation,O
    Agri_Waste,Agri_Process,Includes,deteriorating flora matter,O
    Organic material,Natural_Resource,Includes,decayed plant elements,O
    Agricultural waste,O,Includes,decomposing plant material,O
    organic matter,Natural_Resource,Includes,decaying plant material,O
    irrigation residues,Agri_Waste,Includes,salts,O
    irrigation waste,O,Includes,trace elements,O
    irrigation leftovers,O,Includes,trace metals,O
    farming residues,O,Includes,trace elements,O
    agricultural waste,O,Includes,trace metals,O
    irrigation byproducts,Agri_Waste,Includes,trace elements,O
    irrigation residues,Agri_Waste,Includes,trace metals,O
    irrigation residues,Agri_Waste,Includes,microorganisms,Organism
    Gardening,O,Conjunction,Farming,O
    Horticultural,O,Conjunction,Agricultural,O
    Orcharding,Agri_Process,Conjunction,Agricultural,O
    Floricultural,O,Conjunction,Agricultural,O
    Gardening,O,Conjunction,Farming,O
    horticultural,Agri_Process,Conjunction,Agricultural,O
    Soil pollution,Agri_Pollution,Caused_By,horticultural practices,Agri_Process
    Contaminating farmlands,O,Caused_By,horticulture,Agri_Process
    Soil contamination,Agri_Pollution,Caused_By,horticultural activities,Agri_Process
    Polluting the earth,O,Caused_By,gardening,O
    Soil degradation,Agri_Pollution,Caused_By,horticultural methods,Agri_Process
    contaminating the soils,Agri_Pollution,Caused_By,horticultural,Agri_Process
    soil erosion,O,Caused_By,horticultural practices,Agri_Process
    deposits,O,Caused_By,horticultural activities,Agri_Process
    silt,O,Caused_By,horticultural impact,Agri_Process
    particle sediments,O,Caused_By,horticultural methods,Agri_Process
    sediments,O,Caused_By,horticultural,Agri_Process
    Polluting the land,O,Caused_By,Agricultural,O
    Soil contamination,Agri_Pollution,Caused_By,Agricultural,O
    Contaminating the farmland,O,Caused_By,Agricultural,O
    Land pollution,O,Caused_By,Agricultural,O
    Soil degradation,Agri_Pollution,Caused_By,Agricultural,O
    contaminating the soils,Agri_Pollution,Caused_By,Agricultural,O
    sedimentation,O,Caused_By,Farming practices,O
    deposits,O,Caused_By,Agricultural activities,O
    silt,O,Caused_By,Agriculture,Agri_Process
    alluvium,Crop,Caused_By,Farming,O
    settled particles,O,Caused_By,Agricultural practices,O
    sediments,O,Caused_By,Agricultural,O
    Agri_Pollution,Agri_Process,Includes,natural and synthetic fertilizers,O
    Agricultural pollutants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollution,Agri_Process,Includes,natural and chemical fertilizers,O
    Agricultural contaminants,Agri_Pollution,Includes,natural and chemical fertilizers,O
    Agri_Pollution,Agri_Process,Includes,organic and chemical fertilizers,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollutants,Crop,Includes,insecticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,pest control substances,Agri_Pollution
    Agricultural pollution,O,Includes,herbicides,Chemical
    Agri_Pollutants,Crop,Includes,chemical pesticides,Chemical
    Farming contaminants,Agri_Pollution,Includes,pest management products,O
    Agricultural contaminants,Agri_Pollution,Includes,pesticides,Chemical
    Agri_Pollution,Agri_Process,Includes,weed killers,O
    Agricultural pollutants,Agri_Pollution,Includes,herbicides,Chemical
    Farm pollution,O,Includes,weed control chemicals,Chemical
    Agri_Contaminants,Chemical,Includes,herbicides,Chemical
    Agricultural pollution,O,Includes,weed management agents,Person
    Agricultural contaminants,Agri_Pollution,Includes,herbicides,Chemical
    Agri_Pollution,Agri_Process,Includes,pesticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,insect repellents,Chemical
    Agri_Pollution,Agri_Process,Includes,bug killers,O
    Agricultural contaminants,Agri_Pollution,Includes,pest management chemicals,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,insecticides,Chemical
    farm residue,O,Includes,animal wastes,Agri_Waste
    plant debris,O,Includes,animal wastes,Agri_Waste
    waste material,O,Includes,animal wastes,Agri_Waste
    agricultural waste,O,Includes,animal wastes,Agri_Waste
    biodegradable waste,Agri_Waste,Includes,animal wastes,Agri_Waste
    organic matter,Natural_Resource,Includes,animal wastes,Agri_Waste
    Agri_Residue,Crop,Includes,decomposing botanical substances,O
    Organic debris,Agri_Waste,Includes,rotting vegetation,O
    Agri_Waste,Agri_Process,Includes,deteriorating flora matter,O
    Organic material,Natural_Resource,Includes,decayed plant elements,O
    Agricultural waste,O,Includes,decomposing plant material,O
    organic matter,Natural_Resource,Includes,decaying plant material,O
    irrigation residues,Agri_Waste,Includes,salts,O
    irrigation waste,O,Includes,trace elements,O
    irrigation leftovers,O,Includes,trace metals,O
    farming residues,O,Includes,trace elements,O
    agricultural waste,O,Includes,trace metals,O
    irrigation byproducts,Agri_Waste,Includes,trace elements,O
    irrigation residues,Agri_Waste,Includes,trace metals,O
    irrigation residues,Agri_Waste,Includes,microorganisms,Organism
    Phosphorus,Nutrient,Synonym_Of,P,O
    Gardening,O,Conjunction,Farming,O
    Horticultural,O,Conjunction,Agricultural,O
    Orcharding,Agri_Process,Conjunction,Agricultural,O
    Floricultural,O,Conjunction,Agricultural,O
    Gardening,O,Conjunction,Farming,O
    horticultural,Agri_Process,Conjunction,Agricultural,O
    Soil pollution,Agri_Pollution,Caused_By,horticultural practices,Agri_Process
    Contaminating farmlands,O,Caused_By,horticulture,Agri_Process
    Soil contamination,Agri_Pollution,Caused_By,horticultural activities,Agri_Process
    Polluting the earth,O,Caused_By,gardening,O
    Soil degradation,Agri_Pollution,Caused_By,horticultural methods,Agri_Process
    contaminating the soils,Agri_Pollution,Caused_By,horticultural,Agri_Process
    soil erosion,O,Caused_By,horticultural practices,Agri_Process
    deposits,O,Caused_By,horticultural activities,Agri_Process
    silt,O,Caused_By,horticultural impact,Agri_Process
    particle sediments,O,Caused_By,horticultural methods,Agri_Process
    sediments,O,Caused_By,horticultural,Agri_Process
    Polluting the land,O,Caused_By,Agricultural,O
    Soil contamination,Agri_Pollution,Caused_By,Agricultural,O
    Contaminating the farmland,O,Caused_By,Agricultural,O
    Land pollution,O,Caused_By,Agricultural,O
    Soil degradation,Agri_Pollution,Caused_By,Agricultural,O
    contaminating the soils,Agri_Pollution,Caused_By,Agricultural,O
    sedimentation,O,Caused_By,Farming practices,O
    deposits,O,Caused_By,Agricultural activities,O
    silt,O,Caused_By,Agriculture,Agri_Process
    alluvium,Crop,Caused_By,Farming,O
    settled particles,O,Caused_By,Agricultural practices,O
    sediments,O,Caused_By,Agricultural,O
    Agri_Pollution,Agri_Process,Includes,natural and synthetic fertilizers,O
    Agricultural pollutants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollution,Agri_Process,Includes,natural and chemical fertilizers,O
    Agricultural contaminants,Agri_Pollution,Includes,natural and chemical fertilizers,O
    Agri_Pollution,Agri_Process,Includes,organic and chemical fertilizers,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollutants,Crop,Includes,insecticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,pest control substances,Agri_Pollution
    Agricultural pollution,O,Includes,herbicides,Chemical
    Agri_Pollutants,Crop,Includes,chemical pesticides,Chemical
    Farming contaminants,Agri_Pollution,Includes,pest management products,O
    Agricultural contaminants,Agri_Pollution,Includes,pesticides,Chemical
    Agri_Pollution,Agri_Process,Includes,weed killers,O
    Agricultural pollutants,Agri_Pollution,Includes,herbicides,Chemical
    Farm pollution,O,Includes,weed control chemicals,Chemical
    Agri_Contaminants,Chemical,Includes,herbicides,Chemical
    Agricultural pollution,O,Includes,weed management agents,Person
    Agricultural contaminants,Agri_Pollution,Includes,herbicides,Chemical
    Agri_Pollution,Agri_Process,Includes,pesticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,insect repellents,Chemical
    Agri_Pollution,Agri_Process,Includes,bug killers,O
    Agricultural contaminants,Agri_Pollution,Includes,pest management chemicals,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,insecticides,Chemical
    farm residue,O,Includes,animal wastes,Agri_Waste
    plant debris,O,Includes,animal wastes,Agri_Waste
    waste material,O,Includes,animal wastes,Agri_Waste
    agricultural waste,O,Includes,animal wastes,Agri_Waste
    biodegradable waste,Agri_Waste,Includes,animal wastes,Agri_Waste
    organic matter,Natural_Resource,Includes,animal wastes,Agri_Waste
    Agri_Residue,Crop,Includes,decomposing botanical substances,O
    Organic debris,Agri_Waste,Includes,rotting vegetation,O
    Agri_Waste,Agri_Process,Includes,deteriorating flora matter,O
    Organic material,Natural_Resource,Includes,decayed plant elements,O
    Agricultural waste,O,Includes,decomposing plant material,O
    organic matter,Natural_Resource,Includes,decaying plant material,O
    irrigation residues,Agri_Waste,Includes,salts,O
    irrigation waste,O,Includes,trace elements,O
    irrigation leftovers,O,Includes,trace metals,O
    farming residues,O,Includes,trace elements,O
    agricultural waste,O,Includes,trace metals,O
    irrigation byproducts,Agri_Waste,Includes,trace elements,O
    irrigation residues,Agri_Waste,Includes,trace metals,O
    irrigation residues,Agri_Waste,Includes,microorganisms,Organism
    Phosphorus,Nutrient,Synonym_Of,P,O
    Nutrient,Nutrient,Synonym_Of,Phosphorus,Nutrient
    Essential element,O,Synonym_Of,Phosphorus,Nutrient
    Mineral,Natural_Resource,Synonym_Of,Phosphorus,Nutrient
    Plant nutrient,Nutrient,Synonym_Of,Phosphorus,Nutrient
    P,O,Synonym_Of,Phosphate,Nutrient
    P,O,Synonym_Of,Phosphorus,Nutrient
    Gardening,O,Conjunction,Farming,O
    Horticultural,O,Conjunction,Agricultural,O
    Orcharding,Agri_Process,Conjunction,Agricultural,O
    Floricultural,O,Conjunction,Agricultural,O
    Gardening,O,Conjunction,Farming,O
    horticultural,Agri_Process,Conjunction,Agricultural,O
    Soil pollution,Agri_Pollution,Caused_By,horticultural practices,Agri_Process
    Contaminating farmlands,O,Caused_By,horticulture,Agri_Process
    Soil contamination,Agri_Pollution,Caused_By,horticultural activities,Agri_Process
    Polluting the earth,O,Caused_By,gardening,O
    Soil degradation,Agri_Pollution,Caused_By,horticultural methods,Agri_Process
    contaminating the soils,Agri_Pollution,Caused_By,horticultural,Agri_Process
    soil erosion,O,Caused_By,horticultural practices,Agri_Process
    deposits,O,Caused_By,horticultural activities,Agri_Process
    silt,O,Caused_By,horticultural impact,Agri_Process
    particle sediments,O,Caused_By,horticultural methods,Agri_Process
    sediments,O,Caused_By,horticultural,Agri_Process
    Polluting the land,O,Caused_By,Agricultural,O
    Soil contamination,Agri_Pollution,Caused_By,Agricultural,O
    Contaminating the farmland,O,Caused_By,Agricultural,O
    Land pollution,O,Caused_By,Agricultural,O
    Soil degradation,Agri_Pollution,Caused_By,Agricultural,O
    contaminating the soils,Agri_Pollution,Caused_By,Agricultural,O
    sedimentation,O,Caused_By,Farming practices,O
    deposits,O,Caused_By,Agricultural activities,O
    silt,O,Caused_By,Agriculture,Agri_Process
    alluvium,Crop,Caused_By,Farming,O
    settled particles,O,Caused_By,Agricultural practices,O
    sediments,O,Caused_By,Agricultural,O
    Agri_Pollution,Agri_Process,Includes,natural and synthetic fertilizers,O
    Agricultural pollutants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollution,Agri_Process,Includes,natural and chemical fertilizers,O
    Agricultural contaminants,Agri_Pollution,Includes,natural and chemical fertilizers,O
    Agri_Pollution,Agri_Process,Includes,organic and chemical fertilizers,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollutants,Crop,Includes,insecticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,pest control substances,Agri_Pollution
    Agricultural pollution,O,Includes,herbicides,Chemical
    Agri_Pollutants,Crop,Includes,chemical pesticides,Chemical
    Farming contaminants,Agri_Pollution,Includes,pest management products,O
    Agricultural contaminants,Agri_Pollution,Includes,pesticides,Chemical
    Agri_Pollution,Agri_Process,Includes,weed killers,O
    Agricultural pollutants,Agri_Pollution,Includes,herbicides,Chemical
    Farm pollution,O,Includes,weed control chemicals,Chemical
    Agri_Contaminants,Chemical,Includes,herbicides,Chemical
    Agricultural pollution,O,Includes,weed management agents,Person
    Agricultural contaminants,Agri_Pollution,Includes,herbicides,Chemical
    Agri_Pollution,Agri_Process,Includes,pesticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,insect repellents,Chemical
    Agri_Pollution,Agri_Process,Includes,bug killers,O
    Agricultural contaminants,Agri_Pollution,Includes,pest management chemicals,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,insecticides,Chemical
    farm residue,O,Includes,animal wastes,Agri_Waste
    plant debris,O,Includes,animal wastes,Agri_Waste
    waste material,O,Includes,animal wastes,Agri_Waste
    agricultural waste,O,Includes,animal wastes,Agri_Waste
    biodegradable waste,Agri_Waste,Includes,animal wastes,Agri_Waste
    organic matter,Natural_Resource,Includes,animal wastes,Agri_Waste
    Agri_Residue,Crop,Includes,decomposing botanical substances,O
    Organic debris,Agri_Waste,Includes,rotting vegetation,O
    Agri_Waste,Agri_Process,Includes,deteriorating flora matter,O
    Organic material,Natural_Resource,Includes,decayed plant elements,O
    Agricultural waste,O,Includes,decomposing plant material,O
    organic matter,Natural_Resource,Includes,decaying plant material,O
    irrigation residues,Agri_Waste,Includes,salts,O
    irrigation waste,O,Includes,trace elements,O
    irrigation leftovers,O,Includes,trace metals,O
    farming residues,O,Includes,trace elements,O
    agricultural waste,O,Includes,trace metals,O
    irrigation byproducts,Agri_Waste,Includes,trace elements,O
    irrigation residues,Agri_Waste,Includes,trace metals,O
    irrigation residues,Agri_Waste,Includes,microorganisms,Organism
    Phosphorus,Nutrient,Synonym_Of,P,O
    Nutrient,Nutrient,Synonym_Of,Phosphorus,Nutrient
    Essential element,O,Synonym_Of,Phosphorus,Nutrient
    Mineral,Natural_Resource,Synonym_Of,Phosphorus,Nutrient
    Plant nutrient,Nutrient,Synonym_Of,Phosphorus,Nutrient
    P,O,Synonym_Of,Phosphate,Nutrient
    P,O,Synonym_Of,Phosphorus,Nutrient
    Nitrogenous,O,Helps_In,crop production,O
    Nitrogen element,Nutrient,Helps_In,yield of crops,O
    Nitrogen-rich,Nutrient,Helps_In,agricultural output,O
    Nitrate,Nutrient,Helps_In,crop yield,O
    Nitrogen fertilizer,Nutrient,Helps_In,harvest productivity,O
    nitrogen,Nutrient,Helps_In,crop yield,O
    Gardening,O,Conjunction,Farming,O
    Horticultural,O,Conjunction,Agricultural,O
    Orcharding,Agri_Process,Conjunction,Agricultural,O
    Floricultural,O,Conjunction,Agricultural,O
    Gardening,O,Conjunction,Farming,O
    horticultural,Agri_Process,Conjunction,Agricultural,O
    Soil pollution,Agri_Pollution,Caused_By,horticultural practices,Agri_Process
    Contaminating farmlands,O,Caused_By,horticulture,Agri_Process
    Soil contamination,Agri_Pollution,Caused_By,horticultural activities,Agri_Process
    Polluting the earth,O,Caused_By,gardening,O
    Soil degradation,Agri_Pollution,Caused_By,horticultural methods,Agri_Process
    contaminating the soils,Agri_Pollution,Caused_By,horticultural,Agri_Process
    soil erosion,O,Caused_By,horticultural practices,Agri_Process
    deposits,O,Caused_By,horticultural activities,Agri_Process
    silt,O,Caused_By,horticultural impact,Agri_Process
    particle sediments,O,Caused_By,horticultural methods,Agri_Process
    sediments,O,Caused_By,horticultural,Agri_Process
    Polluting the land,O,Caused_By,Agricultural,O
    Soil contamination,Agri_Pollution,Caused_By,Agricultural,O
    Contaminating the farmland,O,Caused_By,Agricultural,O
    Land pollution,O,Caused_By,Agricultural,O
    Soil degradation,Agri_Pollution,Caused_By,Agricultural,O
    contaminating the soils,Agri_Pollution,Caused_By,Agricultural,O
    sedimentation,O,Caused_By,Farming practices,O
    deposits,O,Caused_By,Agricultural activities,O
    silt,O,Caused_By,Agriculture,Agri_Process
    alluvium,Crop,Caused_By,Farming,O
    settled particles,O,Caused_By,Agricultural practices,O
    sediments,O,Caused_By,Agricultural,O
    Agri_Pollution,Agri_Process,Includes,natural and synthetic fertilizers,O
    Agricultural pollutants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollution,Agri_Process,Includes,natural and chemical fertilizers,O
    Agricultural contaminants,Agri_Pollution,Includes,natural and chemical fertilizers,O
    Agri_Pollution,Agri_Process,Includes,organic and chemical fertilizers,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollutants,Crop,Includes,insecticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,pest control substances,Agri_Pollution
    Agricultural pollution,O,Includes,herbicides,Chemical
    Agri_Pollutants,Crop,Includes,chemical pesticides,Chemical
    Farming contaminants,Agri_Pollution,Includes,pest management products,O
    Agricultural contaminants,Agri_Pollution,Includes,pesticides,Chemical
    Agri_Pollution,Agri_Process,Includes,weed killers,O
    Agricultural pollutants,Agri_Pollution,Includes,herbicides,Chemical
    Farm pollution,O,Includes,weed control chemicals,Chemical
    Agri_Contaminants,Chemical,Includes,herbicides,Chemical
    Agricultural pollution,O,Includes,weed management agents,Person
    Agricultural contaminants,Agri_Pollution,Includes,herbicides,Chemical
    Agri_Pollution,Agri_Process,Includes,pesticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,insect repellents,Chemical
    Agri_Pollution,Agri_Process,Includes,bug killers,O
    Agricultural contaminants,Agri_Pollution,Includes,pest management chemicals,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,insecticides,Chemical
    farm residue,O,Includes,animal wastes,Agri_Waste
    plant debris,O,Includes,animal wastes,Agri_Waste
    waste material,O,Includes,animal wastes,Agri_Waste
    agricultural waste,O,Includes,animal wastes,Agri_Waste
    biodegradable waste,Agri_Waste,Includes,animal wastes,Agri_Waste
    organic matter,Natural_Resource,Includes,animal wastes,Agri_Waste
    Agri_Residue,Crop,Includes,decomposing botanical substances,O
    Organic debris,Agri_Waste,Includes,rotting vegetation,O
    Agri_Waste,Agri_Process,Includes,deteriorating flora matter,O
    Organic material,Natural_Resource,Includes,decayed plant elements,O
    Agricultural waste,O,Includes,decomposing plant material,O
    organic matter,Natural_Resource,Includes,decaying plant material,O
    irrigation residues,Agri_Waste,Includes,salts,O
    irrigation waste,O,Includes,trace elements,O
    irrigation leftovers,O,Includes,trace metals,O
    farming residues,O,Includes,trace elements,O
    agricultural waste,O,Includes,trace metals,O
    irrigation byproducts,Agri_Waste,Includes,trace elements,O
    irrigation residues,Agri_Waste,Includes,trace metals,O
    irrigation residues,Agri_Waste,Includes,microorganisms,Organism
    Phosphorus,Nutrient,Synonym_Of,P,O
    Nutrient,Nutrient,Synonym_Of,Phosphorus,Nutrient
    Essential element,O,Synonym_Of,Phosphorus,Nutrient
    Mineral,Natural_Resource,Synonym_Of,Phosphorus,Nutrient
    Plant nutrient,Nutrient,Synonym_Of,Phosphorus,Nutrient
    P,O,Synonym_Of,Phosphate,Nutrient
    P,O,Synonym_Of,Phosphorus,Nutrient
    Nitrogenous,O,Helps_In,crop production,O
    Nitrogen element,Nutrient,Helps_In,yield of crops,O
    Nitrogen-rich,Nutrient,Helps_In,agricultural output,O
    Nitrate,Nutrient,Helps_In,crop yield,O
    Nitrogen fertilizer,Nutrient,Helps_In,harvest productivity,O
    nitrogen,Nutrient,Helps_In,crop yield,O
    P,O,Helps_In,crop production,O
    Gardening,O,Conjunction,Farming,O
    Horticultural,O,Conjunction,Agricultural,O
    Orcharding,Agri_Process,Conjunction,Agricultural,O
    Floricultural,O,Conjunction,Agricultural,O
    Gardening,O,Conjunction,Farming,O
    horticultural,Agri_Process,Conjunction,Agricultural,O
    Soil pollution,Agri_Pollution,Caused_By,horticultural practices,Agri_Process
    Contaminating farmlands,O,Caused_By,horticulture,Agri_Process
    Soil contamination,Agri_Pollution,Caused_By,horticultural activities,Agri_Process
    Polluting the earth,O,Caused_By,gardening,O
    Soil degradation,Agri_Pollution,Caused_By,horticultural methods,Agri_Process
    contaminating the soils,Agri_Pollution,Caused_By,horticultural,Agri_Process
    soil erosion,O,Caused_By,horticultural practices,Agri_Process
    deposits,O,Caused_By,horticultural activities,Agri_Process
    silt,O,Caused_By,horticultural impact,Agri_Process
    particle sediments,O,Caused_By,horticultural methods,Agri_Process
    sediments,O,Caused_By,horticultural,Agri_Process
    Polluting the land,O,Caused_By,Agricultural,O
    Soil contamination,Agri_Pollution,Caused_By,Agricultural,O
    Contaminating the farmland,O,Caused_By,Agricultural,O
    Land pollution,O,Caused_By,Agricultural,O
    Soil degradation,Agri_Pollution,Caused_By,Agricultural,O
    contaminating the soils,Agri_Pollution,Caused_By,Agricultural,O
    sedimentation,O,Caused_By,Farming practices,O
    deposits,O,Caused_By,Agricultural activities,O
    silt,O,Caused_By,Agriculture,Agri_Process
    alluvium,Crop,Caused_By,Farming,O
    settled particles,O,Caused_By,Agricultural practices,O
    sediments,O,Caused_By,Agricultural,O
    Agri_Pollution,Agri_Process,Includes,natural and synthetic fertilizers,O
    Agricultural pollutants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollution,Agri_Process,Includes,natural and chemical fertilizers,O
    Agricultural contaminants,Agri_Pollution,Includes,natural and chemical fertilizers,O
    Agri_Pollution,Agri_Process,Includes,organic and chemical fertilizers,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollutants,Crop,Includes,insecticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,pest control substances,Agri_Pollution
    Agricultural pollution,O,Includes,herbicides,Chemical
    Agri_Pollutants,Crop,Includes,chemical pesticides,Chemical
    Farming contaminants,Agri_Pollution,Includes,pest management products,O
    Agricultural contaminants,Agri_Pollution,Includes,pesticides,Chemical
    Agri_Pollution,Agri_Process,Includes,weed killers,O
    Agricultural pollutants,Agri_Pollution,Includes,herbicides,Chemical
    Farm pollution,O,Includes,weed control chemicals,Chemical
    Agri_Contaminants,Chemical,Includes,herbicides,Chemical
    Agricultural pollution,O,Includes,weed management agents,Person
    Agricultural contaminants,Agri_Pollution,Includes,herbicides,Chemical
    Agri_Pollution,Agri_Process,Includes,pesticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,insect repellents,Chemical
    Agri_Pollution,Agri_Process,Includes,bug killers,O
    Agricultural contaminants,Agri_Pollution,Includes,pest management chemicals,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,insecticides,Chemical
    farm residue,O,Includes,animal wastes,Agri_Waste
    plant debris,O,Includes,animal wastes,Agri_Waste
    waste material,O,Includes,animal wastes,Agri_Waste
    agricultural waste,O,Includes,animal wastes,Agri_Waste
    biodegradable waste,Agri_Waste,Includes,animal wastes,Agri_Waste
    organic matter,Natural_Resource,Includes,animal wastes,Agri_Waste
    Agri_Residue,Crop,Includes,decomposing botanical substances,O
    Organic debris,Agri_Waste,Includes,rotting vegetation,O
    Agri_Waste,Agri_Process,Includes,deteriorating flora matter,O
    Organic material,Natural_Resource,Includes,decayed plant elements,O
    Agricultural waste,O,Includes,decomposing plant material,O
    organic matter,Natural_Resource,Includes,decaying plant material,O
    irrigation residues,Agri_Waste,Includes,salts,O
    irrigation waste,O,Includes,trace elements,O
    irrigation leftovers,O,Includes,trace metals,O
    farming residues,O,Includes,trace elements,O
    agricultural waste,O,Includes,trace metals,O
    irrigation byproducts,Agri_Waste,Includes,trace elements,O
    irrigation residues,Agri_Waste,Includes,trace metals,O
    irrigation residues,Agri_Waste,Includes,microorganisms,Organism
    Phosphorus,Nutrient,Synonym_Of,P,O
    Nutrient,Nutrient,Synonym_Of,Phosphorus,Nutrient
    Essential element,O,Synonym_Of,Phosphorus,Nutrient
    Mineral,Natural_Resource,Synonym_Of,Phosphorus,Nutrient
    Plant nutrient,Nutrient,Synonym_Of,Phosphorus,Nutrient
    P,O,Synonym_Of,Phosphate,Nutrient
    P,O,Synonym_Of,Phosphorus,Nutrient
    Nitrogenous,O,Helps_In,crop production,O
    Nitrogen element,Nutrient,Helps_In,yield of crops,O
    Nitrogen-rich,Nutrient,Helps_In,agricultural output,O
    Nitrate,Nutrient,Helps_In,crop yield,O
    Nitrogen fertilizer,Nutrient,Helps_In,harvest productivity,O
    nitrogen,Nutrient,Helps_In,crop yield,O
    P,O,Helps_In,crop production,O
    phosphorus,Natural_Resource,Synonym_Of,P,O
    Gardening,O,Conjunction,Farming,O
    Horticultural,O,Conjunction,Agricultural,O
    Orcharding,Agri_Process,Conjunction,Agricultural,O
    Floricultural,O,Conjunction,Agricultural,O
    Gardening,O,Conjunction,Farming,O
    horticultural,Agri_Process,Conjunction,Agricultural,O
    Soil pollution,Agri_Pollution,Caused_By,horticultural practices,Agri_Process
    Contaminating farmlands,O,Caused_By,horticulture,Agri_Process
    Soil contamination,Agri_Pollution,Caused_By,horticultural activities,Agri_Process
    Polluting the earth,O,Caused_By,gardening,O
    Soil degradation,Agri_Pollution,Caused_By,horticultural methods,Agri_Process
    contaminating the soils,Agri_Pollution,Caused_By,horticultural,Agri_Process
    soil erosion,O,Caused_By,horticultural practices,Agri_Process
    deposits,O,Caused_By,horticultural activities,Agri_Process
    silt,O,Caused_By,horticultural impact,Agri_Process
    particle sediments,O,Caused_By,horticultural methods,Agri_Process
    sediments,O,Caused_By,horticultural,Agri_Process
    Polluting the land,O,Caused_By,Agricultural,O
    Soil contamination,Agri_Pollution,Caused_By,Agricultural,O
    Contaminating the farmland,O,Caused_By,Agricultural,O
    Land pollution,O,Caused_By,Agricultural,O
    Soil degradation,Agri_Pollution,Caused_By,Agricultural,O
    contaminating the soils,Agri_Pollution,Caused_By,Agricultural,O
    sedimentation,O,Caused_By,Farming practices,O
    deposits,O,Caused_By,Agricultural activities,O
    silt,O,Caused_By,Agriculture,Agri_Process
    alluvium,Crop,Caused_By,Farming,O
    settled particles,O,Caused_By,Agricultural practices,O
    sediments,O,Caused_By,Agricultural,O
    Agri_Pollution,Agri_Process,Includes,natural and synthetic fertilizers,O
    Agricultural pollutants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollution,Agri_Process,Includes,natural and chemical fertilizers,O
    Agricultural contaminants,Agri_Pollution,Includes,natural and chemical fertilizers,O
    Agri_Pollution,Agri_Process,Includes,organic and chemical fertilizers,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollutants,Crop,Includes,insecticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,pest control substances,Agri_Pollution
    Agricultural pollution,O,Includes,herbicides,Chemical
    Agri_Pollutants,Crop,Includes,chemical pesticides,Chemical
    Farming contaminants,Agri_Pollution,Includes,pest management products,O
    Agricultural contaminants,Agri_Pollution,Includes,pesticides,Chemical
    Agri_Pollution,Agri_Process,Includes,weed killers,O
    Agricultural pollutants,Agri_Pollution,Includes,herbicides,Chemical
    Farm pollution,O,Includes,weed control chemicals,Chemical
    Agri_Contaminants,Chemical,Includes,herbicides,Chemical
    Agricultural pollution,O,Includes,weed management agents,Person
    Agricultural contaminants,Agri_Pollution,Includes,herbicides,Chemical
    Agri_Pollution,Agri_Process,Includes,pesticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,insect repellents,Chemical
    Agri_Pollution,Agri_Process,Includes,bug killers,O
    Agricultural contaminants,Agri_Pollution,Includes,pest management chemicals,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,insecticides,Chemical
    farm residue,O,Includes,animal wastes,Agri_Waste
    plant debris,O,Includes,animal wastes,Agri_Waste
    waste material,O,Includes,animal wastes,Agri_Waste
    agricultural waste,O,Includes,animal wastes,Agri_Waste
    biodegradable waste,Agri_Waste,Includes,animal wastes,Agri_Waste
    organic matter,Natural_Resource,Includes,animal wastes,Agri_Waste
    Agri_Residue,Crop,Includes,decomposing botanical substances,O
    Organic debris,Agri_Waste,Includes,rotting vegetation,O
    Agri_Waste,Agri_Process,Includes,deteriorating flora matter,O
    Organic material,Natural_Resource,Includes,decayed plant elements,O
    Agricultural waste,O,Includes,decomposing plant material,O
    organic matter,Natural_Resource,Includes,decaying plant material,O
    irrigation residues,Agri_Waste,Includes,salts,O
    irrigation waste,O,Includes,trace elements,O
    irrigation leftovers,O,Includes,trace metals,O
    farming residues,O,Includes,trace elements,O
    agricultural waste,O,Includes,trace metals,O
    irrigation byproducts,Agri_Waste,Includes,trace elements,O
    irrigation residues,Agri_Waste,Includes,trace metals,O
    irrigation residues,Agri_Waste,Includes,microorganisms,Organism
    Phosphorus,Nutrient,Synonym_Of,P,O
    Nutrient,Nutrient,Synonym_Of,Phosphorus,Nutrient
    Essential element,O,Synonym_Of,Phosphorus,Nutrient
    Mineral,Natural_Resource,Synonym_Of,Phosphorus,Nutrient
    Plant nutrient,Nutrient,Synonym_Of,Phosphorus,Nutrient
    P,O,Synonym_Of,Phosphate,Nutrient
    P,O,Synonym_Of,Phosphorus,Nutrient
    Nitrogenous,O,Helps_In,crop production,O
    Nitrogen element,Nutrient,Helps_In,yield of crops,O
    Nitrogen-rich,Nutrient,Helps_In,agricultural output,O
    Nitrate,Nutrient,Helps_In,crop yield,O
    Nitrogen fertilizer,Nutrient,Helps_In,harvest productivity,O
    nitrogen,Nutrient,Helps_In,crop yield,O
    P,O,Helps_In,crop production,O
    phosphorus,Natural_Resource,Synonym_Of,P,O
    P,O,Synonym_Of,phosphorus,Natural_Resource
    Gardening,O,Conjunction,Farming,O
    Horticultural,O,Conjunction,Agricultural,O
    Orcharding,Agri_Process,Conjunction,Agricultural,O
    Floricultural,O,Conjunction,Agricultural,O
    Gardening,O,Conjunction,Farming,O
    horticultural,Agri_Process,Conjunction,Agricultural,O
    Soil pollution,Agri_Pollution,Caused_By,horticultural practices,Agri_Process
    Contaminating farmlands,O,Caused_By,horticulture,Agri_Process
    Soil contamination,Agri_Pollution,Caused_By,horticultural activities,Agri_Process
    Polluting the earth,O,Caused_By,gardening,O
    Soil degradation,Agri_Pollution,Caused_By,horticultural methods,Agri_Process
    contaminating the soils,Agri_Pollution,Caused_By,horticultural,Agri_Process
    soil erosion,O,Caused_By,horticultural practices,Agri_Process
    deposits,O,Caused_By,horticultural activities,Agri_Process
    silt,O,Caused_By,horticultural impact,Agri_Process
    particle sediments,O,Caused_By,horticultural methods,Agri_Process
    sediments,O,Caused_By,horticultural,Agri_Process
    Polluting the land,O,Caused_By,Agricultural,O
    Soil contamination,Agri_Pollution,Caused_By,Agricultural,O
    Contaminating the farmland,O,Caused_By,Agricultural,O
    Land pollution,O,Caused_By,Agricultural,O
    Soil degradation,Agri_Pollution,Caused_By,Agricultural,O
    contaminating the soils,Agri_Pollution,Caused_By,Agricultural,O
    sedimentation,O,Caused_By,Farming practices,O
    deposits,O,Caused_By,Agricultural activities,O
    silt,O,Caused_By,Agriculture,Agri_Process
    alluvium,Crop,Caused_By,Farming,O
    settled particles,O,Caused_By,Agricultural practices,O
    sediments,O,Caused_By,Agricultural,O
    Agri_Pollution,Agri_Process,Includes,natural and synthetic fertilizers,O
    Agricultural pollutants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollution,Agri_Process,Includes,natural and chemical fertilizers,O
    Agricultural contaminants,Agri_Pollution,Includes,natural and chemical fertilizers,O
    Agri_Pollution,Agri_Process,Includes,organic and chemical fertilizers,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollutants,Crop,Includes,insecticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,pest control substances,Agri_Pollution
    Agricultural pollution,O,Includes,herbicides,Chemical
    Agri_Pollutants,Crop,Includes,chemical pesticides,Chemical
    Farming contaminants,Agri_Pollution,Includes,pest management products,O
    Agricultural contaminants,Agri_Pollution,Includes,pesticides,Chemical
    Agri_Pollution,Agri_Process,Includes,weed killers,O
    Agricultural pollutants,Agri_Pollution,Includes,herbicides,Chemical
    Farm pollution,O,Includes,weed control chemicals,Chemical
    Agri_Contaminants,Chemical,Includes,herbicides,Chemical
    Agricultural pollution,O,Includes,weed management agents,Person
    Agricultural contaminants,Agri_Pollution,Includes,herbicides,Chemical
    Agri_Pollution,Agri_Process,Includes,pesticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,insect repellents,Chemical
    Agri_Pollution,Agri_Process,Includes,bug killers,O
    Agricultural contaminants,Agri_Pollution,Includes,pest management chemicals,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,insecticides,Chemical
    farm residue,O,Includes,animal wastes,Agri_Waste
    plant debris,O,Includes,animal wastes,Agri_Waste
    waste material,O,Includes,animal wastes,Agri_Waste
    agricultural waste,O,Includes,animal wastes,Agri_Waste
    biodegradable waste,Agri_Waste,Includes,animal wastes,Agri_Waste
    organic matter,Natural_Resource,Includes,animal wastes,Agri_Waste
    Agri_Residue,Crop,Includes,decomposing botanical substances,O
    Organic debris,Agri_Waste,Includes,rotting vegetation,O
    Agri_Waste,Agri_Process,Includes,deteriorating flora matter,O
    Organic material,Natural_Resource,Includes,decayed plant elements,O
    Agricultural waste,O,Includes,decomposing plant material,O
    organic matter,Natural_Resource,Includes,decaying plant material,O
    irrigation residues,Agri_Waste,Includes,salts,O
    irrigation waste,O,Includes,trace elements,O
    irrigation leftovers,O,Includes,trace metals,O
    farming residues,O,Includes,trace elements,O
    agricultural waste,O,Includes,trace metals,O
    irrigation byproducts,Agri_Waste,Includes,trace elements,O
    irrigation residues,Agri_Waste,Includes,trace metals,O
    irrigation residues,Agri_Waste,Includes,microorganisms,Organism
    Phosphorus,Nutrient,Synonym_Of,P,O
    Nutrient,Nutrient,Synonym_Of,Phosphorus,Nutrient
    Essential element,O,Synonym_Of,Phosphorus,Nutrient
    Mineral,Natural_Resource,Synonym_Of,Phosphorus,Nutrient
    Plant nutrient,Nutrient,Synonym_Of,Phosphorus,Nutrient
    P,O,Synonym_Of,Phosphate,Nutrient
    P,O,Synonym_Of,Phosphorus,Nutrient
    Nitrogenous,O,Helps_In,crop production,O
    Nitrogen element,Nutrient,Helps_In,yield of crops,O
    Nitrogen-rich,Nutrient,Helps_In,agricultural output,O
    Nitrate,Nutrient,Helps_In,crop yield,O
    Nitrogen fertilizer,Nutrient,Helps_In,harvest productivity,O
    nitrogen,Nutrient,Helps_In,crop yield,O
    P,O,Helps_In,crop production,O
    phosphorus,Natural_Resource,Synonym_Of,P,O
    P,O,Synonym_Of,phosphorus,Natural_Resource
    Polluting the soil,O,Caused_By,horticulture,Agri_Process
    Soil contamination,Agri_Pollution,Caused_By,horticultural practices,Agri_Process
    Soil pollution,Agri_Pollution,Caused_By,horticulture,Agri_Process
    Contaminating the land,O,Caused_By,horticultural activities,Agri_Process
    Soil adulteration,Agri_Pollution,Caused_By,horticulture,Agri_Process
    contaminating the soils,Agri_Pollution,Caused_By,horticultural,Agri_Process
    Gardening,O,Conjunction,Farming,O
    Horticultural,O,Conjunction,Agricultural,O
    Orcharding,Agri_Process,Conjunction,Agricultural,O
    Floricultural,O,Conjunction,Agricultural,O
    Gardening,O,Conjunction,Farming,O
    horticultural,Agri_Process,Conjunction,Agricultural,O
    Soil pollution,Agri_Pollution,Caused_By,horticultural practices,Agri_Process
    Contaminating farmlands,O,Caused_By,horticulture,Agri_Process
    Soil contamination,Agri_Pollution,Caused_By,horticultural activities,Agri_Process
    Polluting the earth,O,Caused_By,gardening,O
    Soil degradation,Agri_Pollution,Caused_By,horticultural methods,Agri_Process
    contaminating the soils,Agri_Pollution,Caused_By,horticultural,Agri_Process
    soil erosion,O,Caused_By,horticultural practices,Agri_Process
    deposits,O,Caused_By,horticultural activities,Agri_Process
    silt,O,Caused_By,horticultural impact,Agri_Process
    particle sediments,O,Caused_By,horticultural methods,Agri_Process
    sediments,O,Caused_By,horticultural,Agri_Process
    Polluting the land,O,Caused_By,Agricultural,O
    Soil contamination,Agri_Pollution,Caused_By,Agricultural,O
    Contaminating the farmland,O,Caused_By,Agricultural,O
    Land pollution,O,Caused_By,Agricultural,O
    Soil degradation,Agri_Pollution,Caused_By,Agricultural,O
    contaminating the soils,Agri_Pollution,Caused_By,Agricultural,O
    sedimentation,O,Caused_By,Farming practices,O
    deposits,O,Caused_By,Agricultural activities,O
    silt,O,Caused_By,Agriculture,Agri_Process
    alluvium,Crop,Caused_By,Farming,O
    settled particles,O,Caused_By,Agricultural practices,O
    sediments,O,Caused_By,Agricultural,O
    Agri_Pollution,Agri_Process,Includes,natural and synthetic fertilizers,O
    Agricultural pollutants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollution,Agri_Process,Includes,natural and chemical fertilizers,O
    Agricultural contaminants,Agri_Pollution,Includes,natural and chemical fertilizers,O
    Agri_Pollution,Agri_Process,Includes,organic and chemical fertilizers,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollutants,Crop,Includes,insecticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,pest control substances,Agri_Pollution
    Agricultural pollution,O,Includes,herbicides,Chemical
    Agri_Pollutants,Crop,Includes,chemical pesticides,Chemical
    Farming contaminants,Agri_Pollution,Includes,pest management products,O
    Agricultural contaminants,Agri_Pollution,Includes,pesticides,Chemical
    Agri_Pollution,Agri_Process,Includes,weed killers,O
    Agricultural pollutants,Agri_Pollution,Includes,herbicides,Chemical
    Farm pollution,O,Includes,weed control chemicals,Chemical
    Agri_Contaminants,Chemical,Includes,herbicides,Chemical
    Agricultural pollution,O,Includes,weed management agents,Person
    Agricultural contaminants,Agri_Pollution,Includes,herbicides,Chemical
    Agri_Pollution,Agri_Process,Includes,pesticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,insect repellents,Chemical
    Agri_Pollution,Agri_Process,Includes,bug killers,O
    Agricultural contaminants,Agri_Pollution,Includes,pest management chemicals,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,insecticides,Chemical
    farm residue,O,Includes,animal wastes,Agri_Waste
    plant debris,O,Includes,animal wastes,Agri_Waste
    waste material,O,Includes,animal wastes,Agri_Waste
    agricultural waste,O,Includes,animal wastes,Agri_Waste
    biodegradable waste,Agri_Waste,Includes,animal wastes,Agri_Waste
    organic matter,Natural_Resource,Includes,animal wastes,Agri_Waste
    Agri_Residue,Crop,Includes,decomposing botanical substances,O
    Organic debris,Agri_Waste,Includes,rotting vegetation,O
    Agri_Waste,Agri_Process,Includes,deteriorating flora matter,O
    Organic material,Natural_Resource,Includes,decayed plant elements,O
    Agricultural waste,O,Includes,decomposing plant material,O
    organic matter,Natural_Resource,Includes,decaying plant material,O
    irrigation residues,Agri_Waste,Includes,salts,O
    irrigation waste,O,Includes,trace elements,O
    irrigation leftovers,O,Includes,trace metals,O
    farming residues,O,Includes,trace elements,O
    agricultural waste,O,Includes,trace metals,O
    irrigation byproducts,Agri_Waste,Includes,trace elements,O
    irrigation residues,Agri_Waste,Includes,trace metals,O
    irrigation residues,Agri_Waste,Includes,microorganisms,Organism
    Phosphorus,Nutrient,Synonym_Of,P,O
    Nutrient,Nutrient,Synonym_Of,Phosphorus,Nutrient
    Essential element,O,Synonym_Of,Phosphorus,Nutrient
    Mineral,Natural_Resource,Synonym_Of,Phosphorus,Nutrient
    Plant nutrient,Nutrient,Synonym_Of,Phosphorus,Nutrient
    P,O,Synonym_Of,Phosphate,Nutrient
    P,O,Synonym_Of,Phosphorus,Nutrient
    Nitrogenous,O,Helps_In,crop production,O
    Nitrogen element,Nutrient,Helps_In,yield of crops,O
    Nitrogen-rich,Nutrient,Helps_In,agricultural output,O
    Nitrate,Nutrient,Helps_In,crop yield,O
    Nitrogen fertilizer,Nutrient,Helps_In,harvest productivity,O
    nitrogen,Nutrient,Helps_In,crop yield,O
    P,O,Helps_In,crop production,O
    phosphorus,Natural_Resource,Synonym_Of,P,O
    P,O,Synonym_Of,phosphorus,Natural_Resource
    Polluting the soil,O,Caused_By,horticulture,Agri_Process
    Soil contamination,Agri_Pollution,Caused_By,horticultural practices,Agri_Process
    Soil pollution,Agri_Pollution,Caused_By,horticulture,Agri_Process
    Contaminating the land,O,Caused_By,horticultural activities,Agri_Process
    Soil adulteration,Agri_Pollution,Caused_By,horticulture,Agri_Process
    contaminating the soils,Agri_Pollution,Caused_By,horticultural,Agri_Process
    sedimentary deposits,O,Caused_By,horticultural activities,Agri_Process
    sediment,O,Caused_By,horticultural practices,Agri_Process
    sedimentation,O,Caused_By,horticultural impacts,Agri_Process
    sediments,O,Caused_By,horticultural,Agri_Process
    Gardening,O,Conjunction,Farming,O
    Horticultural,O,Conjunction,Agricultural,O
    Orcharding,Agri_Process,Conjunction,Agricultural,O
    Floricultural,O,Conjunction,Agricultural,O
    Gardening,O,Conjunction,Farming,O
    horticultural,Agri_Process,Conjunction,Agricultural,O
    Soil pollution,Agri_Pollution,Caused_By,horticultural practices,Agri_Process
    Contaminating farmlands,O,Caused_By,horticulture,Agri_Process
    Soil contamination,Agri_Pollution,Caused_By,horticultural activities,Agri_Process
    Polluting the earth,O,Caused_By,gardening,O
    Soil degradation,Agri_Pollution,Caused_By,horticultural methods,Agri_Process
    contaminating the soils,Agri_Pollution,Caused_By,horticultural,Agri_Process
    soil erosion,O,Caused_By,horticultural practices,Agri_Process
    deposits,O,Caused_By,horticultural activities,Agri_Process
    silt,O,Caused_By,horticultural impact,Agri_Process
    particle sediments,O,Caused_By,horticultural methods,Agri_Process
    sediments,O,Caused_By,horticultural,Agri_Process
    Polluting the land,O,Caused_By,Agricultural,O
    Soil contamination,Agri_Pollution,Caused_By,Agricultural,O
    Contaminating the farmland,O,Caused_By,Agricultural,O
    Land pollution,O,Caused_By,Agricultural,O
    Soil degradation,Agri_Pollution,Caused_By,Agricultural,O
    contaminating the soils,Agri_Pollution,Caused_By,Agricultural,O
    sedimentation,O,Caused_By,Farming practices,O
    deposits,O,Caused_By,Agricultural activities,O
    silt,O,Caused_By,Agriculture,Agri_Process
    alluvium,Crop,Caused_By,Farming,O
    settled particles,O,Caused_By,Agricultural practices,O
    sediments,O,Caused_By,Agricultural,O
    Agri_Pollution,Agri_Process,Includes,natural and synthetic fertilizers,O
    Agricultural pollutants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollution,Agri_Process,Includes,natural and chemical fertilizers,O
    Agricultural contaminants,Agri_Pollution,Includes,natural and chemical fertilizers,O
    Agri_Pollution,Agri_Process,Includes,organic and chemical fertilizers,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollutants,Crop,Includes,insecticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,pest control substances,Agri_Pollution
    Agricultural pollution,O,Includes,herbicides,Chemical
    Agri_Pollutants,Crop,Includes,chemical pesticides,Chemical
    Farming contaminants,Agri_Pollution,Includes,pest management products,O
    Agricultural contaminants,Agri_Pollution,Includes,pesticides,Chemical
    Agri_Pollution,Agri_Process,Includes,weed killers,O
    Agricultural pollutants,Agri_Pollution,Includes,herbicides,Chemical
    Farm pollution,O,Includes,weed control chemicals,Chemical
    Agri_Contaminants,Chemical,Includes,herbicides,Chemical
    Agricultural pollution,O,Includes,weed management agents,Person
    Agricultural contaminants,Agri_Pollution,Includes,herbicides,Chemical
    Agri_Pollution,Agri_Process,Includes,pesticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,insect repellents,Chemical
    Agri_Pollution,Agri_Process,Includes,bug killers,O
    Agricultural contaminants,Agri_Pollution,Includes,pest management chemicals,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,insecticides,Chemical
    farm residue,O,Includes,animal wastes,Agri_Waste
    plant debris,O,Includes,animal wastes,Agri_Waste
    waste material,O,Includes,animal wastes,Agri_Waste
    agricultural waste,O,Includes,animal wastes,Agri_Waste
    biodegradable waste,Agri_Waste,Includes,animal wastes,Agri_Waste
    organic matter,Natural_Resource,Includes,animal wastes,Agri_Waste
    Agri_Residue,Crop,Includes,decomposing botanical substances,O
    Organic debris,Agri_Waste,Includes,rotting vegetation,O
    Agri_Waste,Agri_Process,Includes,deteriorating flora matter,O
    Organic material,Natural_Resource,Includes,decayed plant elements,O
    Agricultural waste,O,Includes,decomposing plant material,O
    organic matter,Natural_Resource,Includes,decaying plant material,O
    irrigation residues,Agri_Waste,Includes,salts,O
    irrigation waste,O,Includes,trace elements,O
    irrigation leftovers,O,Includes,trace metals,O
    farming residues,O,Includes,trace elements,O
    agricultural waste,O,Includes,trace metals,O
    irrigation byproducts,Agri_Waste,Includes,trace elements,O
    irrigation residues,Agri_Waste,Includes,trace metals,O
    irrigation residues,Agri_Waste,Includes,microorganisms,Organism
    Phosphorus,Nutrient,Synonym_Of,P,O
    Nutrient,Nutrient,Synonym_Of,Phosphorus,Nutrient
    Essential element,O,Synonym_Of,Phosphorus,Nutrient
    Mineral,Natural_Resource,Synonym_Of,Phosphorus,Nutrient
    Plant nutrient,Nutrient,Synonym_Of,Phosphorus,Nutrient
    P,O,Synonym_Of,Phosphate,Nutrient
    P,O,Synonym_Of,Phosphorus,Nutrient
    Nitrogenous,O,Helps_In,crop production,O
    Nitrogen element,Nutrient,Helps_In,yield of crops,O
    Nitrogen-rich,Nutrient,Helps_In,agricultural output,O
    Nitrate,Nutrient,Helps_In,crop yield,O
    Nitrogen fertilizer,Nutrient,Helps_In,harvest productivity,O
    nitrogen,Nutrient,Helps_In,crop yield,O
    P,O,Helps_In,crop production,O
    phosphorus,Natural_Resource,Synonym_Of,P,O
    P,O,Synonym_Of,phosphorus,Natural_Resource
    Polluting the soil,O,Caused_By,horticulture,Agri_Process
    Soil contamination,Agri_Pollution,Caused_By,horticultural practices,Agri_Process
    Soil pollution,Agri_Pollution,Caused_By,horticulture,Agri_Process
    Contaminating the land,O,Caused_By,horticultural activities,Agri_Process
    Soil adulteration,Agri_Pollution,Caused_By,horticulture,Agri_Process
    contaminating the soils,Agri_Pollution,Caused_By,horticultural,Agri_Process
    sedimentary deposits,O,Caused_By,horticultural activities,Agri_Process
    sediment,O,Caused_By,horticultural practices,Agri_Process
    sedimentation,O,Caused_By,horticultural impacts,Agri_Process
    sediments,O,Caused_By,horticultural,Agri_Process
    Polluting the farmland,O,Caused_By,industrial activities,O
    Soil contamination,Agri_Pollution,Caused_By,industrial activities,O
    Tainting the soil,O,Caused_By,industrial activities,O
    Contaminating the cropland,O,Caused_By,industrial activities,O
    Pollution of agricultural soils,O,Caused_By,industrial activities,O
    contaminating the soils,Agri_Pollution,Caused_By,industrial activities,O
    Gardening,O,Conjunction,Farming,O
    Horticultural,O,Conjunction,Agricultural,O
    Orcharding,Agri_Process,Conjunction,Agricultural,O
    Floricultural,O,Conjunction,Agricultural,O
    Gardening,O,Conjunction,Farming,O
    horticultural,Agri_Process,Conjunction,Agricultural,O
    Soil pollution,Agri_Pollution,Caused_By,horticultural practices,Agri_Process
    Contaminating farmlands,O,Caused_By,horticulture,Agri_Process
    Soil contamination,Agri_Pollution,Caused_By,horticultural activities,Agri_Process
    Polluting the earth,O,Caused_By,gardening,O
    Soil degradation,Agri_Pollution,Caused_By,horticultural methods,Agri_Process
    contaminating the soils,Agri_Pollution,Caused_By,horticultural,Agri_Process
    soil erosion,O,Caused_By,horticultural practices,Agri_Process
    deposits,O,Caused_By,horticultural activities,Agri_Process
    silt,O,Caused_By,horticultural impact,Agri_Process
    particle sediments,O,Caused_By,horticultural methods,Agri_Process
    sediments,O,Caused_By,horticultural,Agri_Process
    Polluting the land,O,Caused_By,Agricultural,O
    Soil contamination,Agri_Pollution,Caused_By,Agricultural,O
    Contaminating the farmland,O,Caused_By,Agricultural,O
    Land pollution,O,Caused_By,Agricultural,O
    Soil degradation,Agri_Pollution,Caused_By,Agricultural,O
    contaminating the soils,Agri_Pollution,Caused_By,Agricultural,O
    sedimentation,O,Caused_By,Farming practices,O
    deposits,O,Caused_By,Agricultural activities,O
    silt,O,Caused_By,Agriculture,Agri_Process
    alluvium,Crop,Caused_By,Farming,O
    settled particles,O,Caused_By,Agricultural practices,O
    sediments,O,Caused_By,Agricultural,O
    Agri_Pollution,Agri_Process,Includes,natural and synthetic fertilizers,O
    Agricultural pollutants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollution,Agri_Process,Includes,natural and chemical fertilizers,O
    Agricultural contaminants,Agri_Pollution,Includes,natural and chemical fertilizers,O
    Agri_Pollution,Agri_Process,Includes,organic and chemical fertilizers,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollutants,Crop,Includes,insecticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,pest control substances,Agri_Pollution
    Agricultural pollution,O,Includes,herbicides,Chemical
    Agri_Pollutants,Crop,Includes,chemical pesticides,Chemical
    Farming contaminants,Agri_Pollution,Includes,pest management products,O
    Agricultural contaminants,Agri_Pollution,Includes,pesticides,Chemical
    Agri_Pollution,Agri_Process,Includes,weed killers,O
    Agricultural pollutants,Agri_Pollution,Includes,herbicides,Chemical
    Farm pollution,O,Includes,weed control chemicals,Chemical
    Agri_Contaminants,Chemical,Includes,herbicides,Chemical
    Agricultural pollution,O,Includes,weed management agents,Person
    Agricultural contaminants,Agri_Pollution,Includes,herbicides,Chemical
    Agri_Pollution,Agri_Process,Includes,pesticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,insect repellents,Chemical
    Agri_Pollution,Agri_Process,Includes,bug killers,O
    Agricultural contaminants,Agri_Pollution,Includes,pest management chemicals,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,insecticides,Chemical
    farm residue,O,Includes,animal wastes,Agri_Waste
    plant debris,O,Includes,animal wastes,Agri_Waste
    waste material,O,Includes,animal wastes,Agri_Waste
    agricultural waste,O,Includes,animal wastes,Agri_Waste
    biodegradable waste,Agri_Waste,Includes,animal wastes,Agri_Waste
    organic matter,Natural_Resource,Includes,animal wastes,Agri_Waste
    Agri_Residue,Crop,Includes,decomposing botanical substances,O
    Organic debris,Agri_Waste,Includes,rotting vegetation,O
    Agri_Waste,Agri_Process,Includes,deteriorating flora matter,O
    Organic material,Natural_Resource,Includes,decayed plant elements,O
    Agricultural waste,O,Includes,decomposing plant material,O
    organic matter,Natural_Resource,Includes,decaying plant material,O
    irrigation residues,Agri_Waste,Includes,salts,O
    irrigation waste,O,Includes,trace elements,O
    irrigation leftovers,O,Includes,trace metals,O
    farming residues,O,Includes,trace elements,O
    agricultural waste,O,Includes,trace metals,O
    irrigation byproducts,Agri_Waste,Includes,trace elements,O
    irrigation residues,Agri_Waste,Includes,trace metals,O
    irrigation residues,Agri_Waste,Includes,microorganisms,Organism
    Phosphorus,Nutrient,Synonym_Of,P,O
    Nutrient,Nutrient,Synonym_Of,Phosphorus,Nutrient
    Essential element,O,Synonym_Of,Phosphorus,Nutrient
    Mineral,Natural_Resource,Synonym_Of,Phosphorus,Nutrient
    Plant nutrient,Nutrient,Synonym_Of,Phosphorus,Nutrient
    P,O,Synonym_Of,Phosphate,Nutrient
    P,O,Synonym_Of,Phosphorus,Nutrient
    Nitrogenous,O,Helps_In,crop production,O
    Nitrogen element,Nutrient,Helps_In,yield of crops,O
    Nitrogen-rich,Nutrient,Helps_In,agricultural output,O
    Nitrate,Nutrient,Helps_In,crop yield,O
    Nitrogen fertilizer,Nutrient,Helps_In,harvest productivity,O
    nitrogen,Nutrient,Helps_In,crop yield,O
    P,O,Helps_In,crop production,O
    phosphorus,Natural_Resource,Synonym_Of,P,O
    P,O,Synonym_Of,phosphorus,Natural_Resource
    Polluting the soil,O,Caused_By,horticulture,Agri_Process
    Soil contamination,Agri_Pollution,Caused_By,horticultural practices,Agri_Process
    Soil pollution,Agri_Pollution,Caused_By,horticulture,Agri_Process
    Contaminating the land,O,Caused_By,horticultural activities,Agri_Process
    Soil adulteration,Agri_Pollution,Caused_By,horticulture,Agri_Process
    contaminating the soils,Agri_Pollution,Caused_By,horticultural,Agri_Process
    sedimentary deposits,O,Caused_By,horticultural activities,Agri_Process
    sediment,O,Caused_By,horticultural practices,Agri_Process
    sedimentation,O,Caused_By,horticultural impacts,Agri_Process
    sediments,O,Caused_By,horticultural,Agri_Process
    Polluting the farmland,O,Caused_By,industrial activities,O
    Soil contamination,Agri_Pollution,Caused_By,industrial activities,O
    Tainting the soil,O,Caused_By,industrial activities,O
    Contaminating the cropland,O,Caused_By,industrial activities,O
    Pollution of agricultural soils,O,Caused_By,industrial activities,O
    contaminating the soils,Agri_Pollution,Caused_By,industrial activities,O
    Agri_Pollution,Agri_Process,Caused_By,manufacturing operations,O
    Sedimentary deposits,O,Caused_By,industrial processes,O
    Agri_Pollutant,Crop,Caused_By,industrial activities,O
    Sediment,O,Caused_By,industrial operations,O
    Agri_Pollution,Agri_Process,Caused_By,industrial practices,O
    sediments,O,Caused_By,industrial activities,O
    Gardening,O,Conjunction,Farming,O
    Horticultural,O,Conjunction,Agricultural,O
    Orcharding,Agri_Process,Conjunction,Agricultural,O
    Floricultural,O,Conjunction,Agricultural,O
    Gardening,O,Conjunction,Farming,O
    horticultural,Agri_Process,Conjunction,Agricultural,O
    Soil pollution,Agri_Pollution,Caused_By,horticultural practices,Agri_Process
    Contaminating farmlands,O,Caused_By,horticulture,Agri_Process
    Soil contamination,Agri_Pollution,Caused_By,horticultural activities,Agri_Process
    Polluting the earth,O,Caused_By,gardening,O
    Soil degradation,Agri_Pollution,Caused_By,horticultural methods,Agri_Process
    contaminating the soils,Agri_Pollution,Caused_By,horticultural,Agri_Process
    soil erosion,O,Caused_By,horticultural practices,Agri_Process
    deposits,O,Caused_By,horticultural activities,Agri_Process
    silt,O,Caused_By,horticultural impact,Agri_Process
    particle sediments,O,Caused_By,horticultural methods,Agri_Process
    sediments,O,Caused_By,horticultural,Agri_Process
    Polluting the land,O,Caused_By,Agricultural,O
    Soil contamination,Agri_Pollution,Caused_By,Agricultural,O
    Contaminating the farmland,O,Caused_By,Agricultural,O
    Land pollution,O,Caused_By,Agricultural,O
    Soil degradation,Agri_Pollution,Caused_By,Agricultural,O
    contaminating the soils,Agri_Pollution,Caused_By,Agricultural,O
    sedimentation,O,Caused_By,Farming practices,O
    deposits,O,Caused_By,Agricultural activities,O
    silt,O,Caused_By,Agriculture,Agri_Process
    alluvium,Crop,Caused_By,Farming,O
    settled particles,O,Caused_By,Agricultural practices,O
    sediments,O,Caused_By,Agricultural,O
    Agri_Pollution,Agri_Process,Includes,natural and synthetic fertilizers,O
    Agricultural pollutants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollution,Agri_Process,Includes,natural and chemical fertilizers,O
    Agricultural contaminants,Agri_Pollution,Includes,natural and chemical fertilizers,O
    Agri_Pollution,Agri_Process,Includes,organic and chemical fertilizers,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollutants,Crop,Includes,insecticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,pest control substances,Agri_Pollution
    Agricultural pollution,O,Includes,herbicides,Chemical
    Agri_Pollutants,Crop,Includes,chemical pesticides,Chemical
    Farming contaminants,Agri_Pollution,Includes,pest management products,O
    Agricultural contaminants,Agri_Pollution,Includes,pesticides,Chemical
    Agri_Pollution,Agri_Process,Includes,weed killers,O
    Agricultural pollutants,Agri_Pollution,Includes,herbicides,Chemical
    Farm pollution,O,Includes,weed control chemicals,Chemical
    Agri_Contaminants,Chemical,Includes,herbicides,Chemical
    Agricultural pollution,O,Includes,weed management agents,Person
    Agricultural contaminants,Agri_Pollution,Includes,herbicides,Chemical
    Agri_Pollution,Agri_Process,Includes,pesticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,insect repellents,Chemical
    Agri_Pollution,Agri_Process,Includes,bug killers,O
    Agricultural contaminants,Agri_Pollution,Includes,pest management chemicals,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,insecticides,Chemical
    farm residue,O,Includes,animal wastes,Agri_Waste
    plant debris,O,Includes,animal wastes,Agri_Waste
    waste material,O,Includes,animal wastes,Agri_Waste
    agricultural waste,O,Includes,animal wastes,Agri_Waste
    biodegradable waste,Agri_Waste,Includes,animal wastes,Agri_Waste
    organic matter,Natural_Resource,Includes,animal wastes,Agri_Waste
    Agri_Residue,Crop,Includes,decomposing botanical substances,O
    Organic debris,Agri_Waste,Includes,rotting vegetation,O
    Agri_Waste,Agri_Process,Includes,deteriorating flora matter,O
    Organic material,Natural_Resource,Includes,decayed plant elements,O
    Agricultural waste,O,Includes,decomposing plant material,O
    organic matter,Natural_Resource,Includes,decaying plant material,O
    irrigation residues,Agri_Waste,Includes,salts,O
    irrigation waste,O,Includes,trace elements,O
    irrigation leftovers,O,Includes,trace metals,O
    farming residues,O,Includes,trace elements,O
    agricultural waste,O,Includes,trace metals,O
    irrigation byproducts,Agri_Waste,Includes,trace elements,O
    irrigation residues,Agri_Waste,Includes,trace metals,O
    irrigation residues,Agri_Waste,Includes,microorganisms,Organism
    Phosphorus,Nutrient,Synonym_Of,P,O
    Nutrient,Nutrient,Synonym_Of,Phosphorus,Nutrient
    Essential element,O,Synonym_Of,Phosphorus,Nutrient
    Mineral,Natural_Resource,Synonym_Of,Phosphorus,Nutrient
    Plant nutrient,Nutrient,Synonym_Of,Phosphorus,Nutrient
    P,O,Synonym_Of,Phosphate,Nutrient
    P,O,Synonym_Of,Phosphorus,Nutrient
    Nitrogenous,O,Helps_In,crop production,O
    Nitrogen element,Nutrient,Helps_In,yield of crops,O
    Nitrogen-rich,Nutrient,Helps_In,agricultural output,O
    Nitrate,Nutrient,Helps_In,crop yield,O
    Nitrogen fertilizer,Nutrient,Helps_In,harvest productivity,O
    nitrogen,Nutrient,Helps_In,crop yield,O
    P,O,Helps_In,crop production,O
    phosphorus,Natural_Resource,Synonym_Of,P,O
    P,O,Synonym_Of,phosphorus,Natural_Resource
    Polluting the soil,O,Caused_By,horticulture,Agri_Process
    Soil contamination,Agri_Pollution,Caused_By,horticultural practices,Agri_Process
    Soil pollution,Agri_Pollution,Caused_By,horticulture,Agri_Process
    Contaminating the land,O,Caused_By,horticultural activities,Agri_Process
    Soil adulteration,Agri_Pollution,Caused_By,horticulture,Agri_Process
    contaminating the soils,Agri_Pollution,Caused_By,horticultural,Agri_Process
    sedimentary deposits,O,Caused_By,horticultural activities,Agri_Process
    sediment,O,Caused_By,horticultural practices,Agri_Process
    sedimentation,O,Caused_By,horticultural impacts,Agri_Process
    sediments,O,Caused_By,horticultural,Agri_Process
    Polluting the farmland,O,Caused_By,industrial activities,O
    Soil contamination,Agri_Pollution,Caused_By,industrial activities,O
    Tainting the soil,O,Caused_By,industrial activities,O
    Contaminating the cropland,O,Caused_By,industrial activities,O
    Pollution of agricultural soils,O,Caused_By,industrial activities,O
    contaminating the soils,Agri_Pollution,Caused_By,industrial activities,O
    Agri_Pollution,Agri_Process,Caused_By,manufacturing operations,O
    Sedimentary deposits,O,Caused_By,industrial processes,O
    Agri_Pollutant,Crop,Caused_By,industrial activities,O
    Sediment,O,Caused_By,industrial operations,O
    Agri_Pollution,Agri_Process,Caused_By,industrial practices,O
    sediments,O,Caused_By,industrial activities,O
    Agri_Pollution,Agri_Process,Includes,natural and synthetic fertilizers,O
    Agricultural pollutants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollutants,Crop,Includes,natural and synthetic fertilizers,O
    Farm pollutants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agricultural pollution,O,Includes,natural and synthetic fertilizers,O
    Agricultural contaminants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Gardening,O,Conjunction,Farming,O
    Horticultural,O,Conjunction,Agricultural,O
    Orcharding,Agri_Process,Conjunction,Agricultural,O
    Floricultural,O,Conjunction,Agricultural,O
    Gardening,O,Conjunction,Farming,O
    horticultural,Agri_Process,Conjunction,Agricultural,O
    Soil pollution,Agri_Pollution,Caused_By,horticultural practices,Agri_Process
    Contaminating farmlands,O,Caused_By,horticulture,Agri_Process
    Soil contamination,Agri_Pollution,Caused_By,horticultural activities,Agri_Process
    Polluting the earth,O,Caused_By,gardening,O
    Soil degradation,Agri_Pollution,Caused_By,horticultural methods,Agri_Process
    contaminating the soils,Agri_Pollution,Caused_By,horticultural,Agri_Process
    soil erosion,O,Caused_By,horticultural practices,Agri_Process
    deposits,O,Caused_By,horticultural activities,Agri_Process
    silt,O,Caused_By,horticultural impact,Agri_Process
    particle sediments,O,Caused_By,horticultural methods,Agri_Process
    sediments,O,Caused_By,horticultural,Agri_Process
    Polluting the land,O,Caused_By,Agricultural,O
    Soil contamination,Agri_Pollution,Caused_By,Agricultural,O
    Contaminating the farmland,O,Caused_By,Agricultural,O
    Land pollution,O,Caused_By,Agricultural,O
    Soil degradation,Agri_Pollution,Caused_By,Agricultural,O
    contaminating the soils,Agri_Pollution,Caused_By,Agricultural,O
    sedimentation,O,Caused_By,Farming practices,O
    deposits,O,Caused_By,Agricultural activities,O
    silt,O,Caused_By,Agriculture,Agri_Process
    alluvium,Crop,Caused_By,Farming,O
    settled particles,O,Caused_By,Agricultural practices,O
    sediments,O,Caused_By,Agricultural,O
    Agri_Pollution,Agri_Process,Includes,natural and synthetic fertilizers,O
    Agricultural pollutants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollution,Agri_Process,Includes,natural and chemical fertilizers,O
    Agricultural contaminants,Agri_Pollution,Includes,natural and chemical fertilizers,O
    Agri_Pollution,Agri_Process,Includes,organic and chemical fertilizers,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollutants,Crop,Includes,insecticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,pest control substances,Agri_Pollution
    Agricultural pollution,O,Includes,herbicides,Chemical
    Agri_Pollutants,Crop,Includes,chemical pesticides,Chemical
    Farming contaminants,Agri_Pollution,Includes,pest management products,O
    Agricultural contaminants,Agri_Pollution,Includes,pesticides,Chemical
    Agri_Pollution,Agri_Process,Includes,weed killers,O
    Agricultural pollutants,Agri_Pollution,Includes,herbicides,Chemical
    Farm pollution,O,Includes,weed control chemicals,Chemical
    Agri_Contaminants,Chemical,Includes,herbicides,Chemical
    Agricultural pollution,O,Includes,weed management agents,Person
    Agricultural contaminants,Agri_Pollution,Includes,herbicides,Chemical
    Agri_Pollution,Agri_Process,Includes,pesticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,insect repellents,Chemical
    Agri_Pollution,Agri_Process,Includes,bug killers,O
    Agricultural contaminants,Agri_Pollution,Includes,pest management chemicals,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,insecticides,Chemical
    farm residue,O,Includes,animal wastes,Agri_Waste
    plant debris,O,Includes,animal wastes,Agri_Waste
    waste material,O,Includes,animal wastes,Agri_Waste
    agricultural waste,O,Includes,animal wastes,Agri_Waste
    biodegradable waste,Agri_Waste,Includes,animal wastes,Agri_Waste
    organic matter,Natural_Resource,Includes,animal wastes,Agri_Waste
    Agri_Residue,Crop,Includes,decomposing botanical substances,O
    Organic debris,Agri_Waste,Includes,rotting vegetation,O
    Agri_Waste,Agri_Process,Includes,deteriorating flora matter,O
    Organic material,Natural_Resource,Includes,decayed plant elements,O
    Agricultural waste,O,Includes,decomposing plant material,O
    organic matter,Natural_Resource,Includes,decaying plant material,O
    irrigation residues,Agri_Waste,Includes,salts,O
    irrigation waste,O,Includes,trace elements,O
    irrigation leftovers,O,Includes,trace metals,O
    farming residues,O,Includes,trace elements,O
    agricultural waste,O,Includes,trace metals,O
    irrigation byproducts,Agri_Waste,Includes,trace elements,O
    irrigation residues,Agri_Waste,Includes,trace metals,O
    irrigation residues,Agri_Waste,Includes,microorganisms,Organism
    Phosphorus,Nutrient,Synonym_Of,P,O
    Nutrient,Nutrient,Synonym_Of,Phosphorus,Nutrient
    Essential element,O,Synonym_Of,Phosphorus,Nutrient
    Mineral,Natural_Resource,Synonym_Of,Phosphorus,Nutrient
    Plant nutrient,Nutrient,Synonym_Of,Phosphorus,Nutrient
    P,O,Synonym_Of,Phosphate,Nutrient
    P,O,Synonym_Of,Phosphorus,Nutrient
    Nitrogenous,O,Helps_In,crop production,O
    Nitrogen element,Nutrient,Helps_In,yield of crops,O
    Nitrogen-rich,Nutrient,Helps_In,agricultural output,O
    Nitrate,Nutrient,Helps_In,crop yield,O
    Nitrogen fertilizer,Nutrient,Helps_In,harvest productivity,O
    nitrogen,Nutrient,Helps_In,crop yield,O
    P,O,Helps_In,crop production,O
    phosphorus,Natural_Resource,Synonym_Of,P,O
    P,O,Synonym_Of,phosphorus,Natural_Resource
    Polluting the soil,O,Caused_By,horticulture,Agri_Process
    Soil contamination,Agri_Pollution,Caused_By,horticultural practices,Agri_Process
    Soil pollution,Agri_Pollution,Caused_By,horticulture,Agri_Process
    Contaminating the land,O,Caused_By,horticultural activities,Agri_Process
    Soil adulteration,Agri_Pollution,Caused_By,horticulture,Agri_Process
    contaminating the soils,Agri_Pollution,Caused_By,horticultural,Agri_Process
    sedimentary deposits,O,Caused_By,horticultural activities,Agri_Process
    sediment,O,Caused_By,horticultural practices,Agri_Process
    sedimentation,O,Caused_By,horticultural impacts,Agri_Process
    sediments,O,Caused_By,horticultural,Agri_Process
    Polluting the farmland,O,Caused_By,industrial activities,O
    Soil contamination,Agri_Pollution,Caused_By,industrial activities,O
    Tainting the soil,O,Caused_By,industrial activities,O
    Contaminating the cropland,O,Caused_By,industrial activities,O
    Pollution of agricultural soils,O,Caused_By,industrial activities,O
    contaminating the soils,Agri_Pollution,Caused_By,industrial activities,O
    Agri_Pollution,Agri_Process,Caused_By,manufacturing operations,O
    Sedimentary deposits,O,Caused_By,industrial processes,O
    Agri_Pollutant,Crop,Caused_By,industrial activities,O
    Sediment,O,Caused_By,industrial operations,O
    Agri_Pollution,Agri_Process,Caused_By,industrial practices,O
    sediments,O,Caused_By,industrial activities,O
    Agri_Pollution,Agri_Process,Includes,natural and synthetic fertilizers,O
    Agricultural pollutants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollutants,Crop,Includes,natural and synthetic fertilizers,O
    Farm pollutants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agricultural pollution,O,Includes,natural and synthetic fertilizers,O
    Agricultural contaminants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollution,Agri_Process,Includes,insecticides,Chemical
    Agricultural pollutants,Agri_Pollution,Includes,pesticides,Chemical
    Farming pollution,O,Includes,pest control substances,Agri_Pollution
    Agri_Contaminants,Chemical,Includes,chemical pesticides,Chemical
    Agricultural pollution,O,Includes,herbicides,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,pesticides,Chemical
    Gardening,O,Conjunction,Farming,O
    Horticultural,O,Conjunction,Agricultural,O
    Orcharding,Agri_Process,Conjunction,Agricultural,O
    Floricultural,O,Conjunction,Agricultural,O
    Gardening,O,Conjunction,Farming,O
    horticultural,Agri_Process,Conjunction,Agricultural,O
    Soil pollution,Agri_Pollution,Caused_By,horticultural practices,Agri_Process
    Contaminating farmlands,O,Caused_By,horticulture,Agri_Process
    Soil contamination,Agri_Pollution,Caused_By,horticultural activities,Agri_Process
    Polluting the earth,O,Caused_By,gardening,O
    Soil degradation,Agri_Pollution,Caused_By,horticultural methods,Agri_Process
    contaminating the soils,Agri_Pollution,Caused_By,horticultural,Agri_Process
    soil erosion,O,Caused_By,horticultural practices,Agri_Process
    deposits,O,Caused_By,horticultural activities,Agri_Process
    silt,O,Caused_By,horticultural impact,Agri_Process
    particle sediments,O,Caused_By,horticultural methods,Agri_Process
    sediments,O,Caused_By,horticultural,Agri_Process
    Polluting the land,O,Caused_By,Agricultural,O
    Soil contamination,Agri_Pollution,Caused_By,Agricultural,O
    Contaminating the farmland,O,Caused_By,Agricultural,O
    Land pollution,O,Caused_By,Agricultural,O
    Soil degradation,Agri_Pollution,Caused_By,Agricultural,O
    contaminating the soils,Agri_Pollution,Caused_By,Agricultural,O
    sedimentation,O,Caused_By,Farming practices,O
    deposits,O,Caused_By,Agricultural activities,O
    silt,O,Caused_By,Agriculture,Agri_Process
    alluvium,Crop,Caused_By,Farming,O
    settled particles,O,Caused_By,Agricultural practices,O
    sediments,O,Caused_By,Agricultural,O
    Agri_Pollution,Agri_Process,Includes,natural and synthetic fertilizers,O
    Agricultural pollutants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollution,Agri_Process,Includes,natural and chemical fertilizers,O
    Agricultural contaminants,Agri_Pollution,Includes,natural and chemical fertilizers,O
    Agri_Pollution,Agri_Process,Includes,organic and chemical fertilizers,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollutants,Crop,Includes,insecticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,pest control substances,Agri_Pollution
    Agricultural pollution,O,Includes,herbicides,Chemical
    Agri_Pollutants,Crop,Includes,chemical pesticides,Chemical
    Farming contaminants,Agri_Pollution,Includes,pest management products,O
    Agricultural contaminants,Agri_Pollution,Includes,pesticides,Chemical
    Agri_Pollution,Agri_Process,Includes,weed killers,O
    Agricultural pollutants,Agri_Pollution,Includes,herbicides,Chemical
    Farm pollution,O,Includes,weed control chemicals,Chemical
    Agri_Contaminants,Chemical,Includes,herbicides,Chemical
    Agricultural pollution,O,Includes,weed management agents,Person
    Agricultural contaminants,Agri_Pollution,Includes,herbicides,Chemical
    Agri_Pollution,Agri_Process,Includes,pesticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,insect repellents,Chemical
    Agri_Pollution,Agri_Process,Includes,bug killers,O
    Agricultural contaminants,Agri_Pollution,Includes,pest management chemicals,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,insecticides,Chemical
    farm residue,O,Includes,animal wastes,Agri_Waste
    plant debris,O,Includes,animal wastes,Agri_Waste
    waste material,O,Includes,animal wastes,Agri_Waste
    agricultural waste,O,Includes,animal wastes,Agri_Waste
    biodegradable waste,Agri_Waste,Includes,animal wastes,Agri_Waste
    organic matter,Natural_Resource,Includes,animal wastes,Agri_Waste
    Agri_Residue,Crop,Includes,decomposing botanical substances,O
    Organic debris,Agri_Waste,Includes,rotting vegetation,O
    Agri_Waste,Agri_Process,Includes,deteriorating flora matter,O
    Organic material,Natural_Resource,Includes,decayed plant elements,O
    Agricultural waste,O,Includes,decomposing plant material,O
    organic matter,Natural_Resource,Includes,decaying plant material,O
    irrigation residues,Agri_Waste,Includes,salts,O
    irrigation waste,O,Includes,trace elements,O
    irrigation leftovers,O,Includes,trace metals,O
    farming residues,O,Includes,trace elements,O
    agricultural waste,O,Includes,trace metals,O
    irrigation byproducts,Agri_Waste,Includes,trace elements,O
    irrigation residues,Agri_Waste,Includes,trace metals,O
    irrigation residues,Agri_Waste,Includes,microorganisms,Organism
    Phosphorus,Nutrient,Synonym_Of,P,O
    Nutrient,Nutrient,Synonym_Of,Phosphorus,Nutrient
    Essential element,O,Synonym_Of,Phosphorus,Nutrient
    Mineral,Natural_Resource,Synonym_Of,Phosphorus,Nutrient
    Plant nutrient,Nutrient,Synonym_Of,Phosphorus,Nutrient
    P,O,Synonym_Of,Phosphate,Nutrient
    P,O,Synonym_Of,Phosphorus,Nutrient
    Nitrogenous,O,Helps_In,crop production,O
    Nitrogen element,Nutrient,Helps_In,yield of crops,O
    Nitrogen-rich,Nutrient,Helps_In,agricultural output,O
    Nitrate,Nutrient,Helps_In,crop yield,O
    Nitrogen fertilizer,Nutrient,Helps_In,harvest productivity,O
    nitrogen,Nutrient,Helps_In,crop yield,O
    P,O,Helps_In,crop production,O
    phosphorus,Natural_Resource,Synonym_Of,P,O
    P,O,Synonym_Of,phosphorus,Natural_Resource
    Polluting the soil,O,Caused_By,horticulture,Agri_Process
    Soil contamination,Agri_Pollution,Caused_By,horticultural practices,Agri_Process
    Soil pollution,Agri_Pollution,Caused_By,horticulture,Agri_Process
    Contaminating the land,O,Caused_By,horticultural activities,Agri_Process
    Soil adulteration,Agri_Pollution,Caused_By,horticulture,Agri_Process
    contaminating the soils,Agri_Pollution,Caused_By,horticultural,Agri_Process
    sedimentary deposits,O,Caused_By,horticultural activities,Agri_Process
    sediment,O,Caused_By,horticultural practices,Agri_Process
    sedimentation,O,Caused_By,horticultural impacts,Agri_Process
    sediments,O,Caused_By,horticultural,Agri_Process
    Polluting the farmland,O,Caused_By,industrial activities,O
    Soil contamination,Agri_Pollution,Caused_By,industrial activities,O
    Tainting the soil,O,Caused_By,industrial activities,O
    Contaminating the cropland,O,Caused_By,industrial activities,O
    Pollution of agricultural soils,O,Caused_By,industrial activities,O
    contaminating the soils,Agri_Pollution,Caused_By,industrial activities,O
    Agri_Pollution,Agri_Process,Caused_By,manufacturing operations,O
    Sedimentary deposits,O,Caused_By,industrial processes,O
    Agri_Pollutant,Crop,Caused_By,industrial activities,O
    Sediment,O,Caused_By,industrial operations,O
    Agri_Pollution,Agri_Process,Caused_By,industrial practices,O
    sediments,O,Caused_By,industrial activities,O
    Agri_Pollution,Agri_Process,Includes,natural and synthetic fertilizers,O
    Agricultural pollutants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollutants,Crop,Includes,natural and synthetic fertilizers,O
    Farm pollutants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agricultural pollution,O,Includes,natural and synthetic fertilizers,O
    Agricultural contaminants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollution,Agri_Process,Includes,insecticides,Chemical
    Agricultural pollutants,Agri_Pollution,Includes,pesticides,Chemical
    Farming pollution,O,Includes,pest control substances,Agri_Pollution
    Agri_Contaminants,Chemical,Includes,chemical pesticides,Chemical
    Agricultural pollution,O,Includes,herbicides,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,pesticides,Chemical
    Agri_Pollutants,Crop,Includes,weed killers,O
    Farming pollutants,Agri_Pollution,Includes,herbicides,Chemical
    Agricultural pollution,O,Includes,herbicides,Chemical
    Agri_Contaminants,Chemical,Includes,weed killers,O
    Farm pollutants,Agri_Pollution,Includes,herbicides,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,herbicides,Chemical
    Gardening,O,Conjunction,Farming,O
    Horticultural,O,Conjunction,Agricultural,O
    Orcharding,Agri_Process,Conjunction,Agricultural,O
    Floricultural,O,Conjunction,Agricultural,O
    Gardening,O,Conjunction,Farming,O
    horticultural,Agri_Process,Conjunction,Agricultural,O
    Soil pollution,Agri_Pollution,Caused_By,horticultural practices,Agri_Process
    Contaminating farmlands,O,Caused_By,horticulture,Agri_Process
    Soil contamination,Agri_Pollution,Caused_By,horticultural activities,Agri_Process
    Polluting the earth,O,Caused_By,gardening,O
    Soil degradation,Agri_Pollution,Caused_By,horticultural methods,Agri_Process
    contaminating the soils,Agri_Pollution,Caused_By,horticultural,Agri_Process
    soil erosion,O,Caused_By,horticultural practices,Agri_Process
    deposits,O,Caused_By,horticultural activities,Agri_Process
    silt,O,Caused_By,horticultural impact,Agri_Process
    particle sediments,O,Caused_By,horticultural methods,Agri_Process
    sediments,O,Caused_By,horticultural,Agri_Process
    Polluting the land,O,Caused_By,Agricultural,O
    Soil contamination,Agri_Pollution,Caused_By,Agricultural,O
    Contaminating the farmland,O,Caused_By,Agricultural,O
    Land pollution,O,Caused_By,Agricultural,O
    Soil degradation,Agri_Pollution,Caused_By,Agricultural,O
    contaminating the soils,Agri_Pollution,Caused_By,Agricultural,O
    sedimentation,O,Caused_By,Farming practices,O
    deposits,O,Caused_By,Agricultural activities,O
    silt,O,Caused_By,Agriculture,Agri_Process
    alluvium,Crop,Caused_By,Farming,O
    settled particles,O,Caused_By,Agricultural practices,O
    sediments,O,Caused_By,Agricultural,O
    Agri_Pollution,Agri_Process,Includes,natural and synthetic fertilizers,O
    Agricultural pollutants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollution,Agri_Process,Includes,natural and chemical fertilizers,O
    Agricultural contaminants,Agri_Pollution,Includes,natural and chemical fertilizers,O
    Agri_Pollution,Agri_Process,Includes,organic and chemical fertilizers,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollutants,Crop,Includes,insecticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,pest control substances,Agri_Pollution
    Agricultural pollution,O,Includes,herbicides,Chemical
    Agri_Pollutants,Crop,Includes,chemical pesticides,Chemical
    Farming contaminants,Agri_Pollution,Includes,pest management products,O
    Agricultural contaminants,Agri_Pollution,Includes,pesticides,Chemical
    Agri_Pollution,Agri_Process,Includes,weed killers,O
    Agricultural pollutants,Agri_Pollution,Includes,herbicides,Chemical
    Farm pollution,O,Includes,weed control chemicals,Chemical
    Agri_Contaminants,Chemical,Includes,herbicides,Chemical
    Agricultural pollution,O,Includes,weed management agents,Person
    Agricultural contaminants,Agri_Pollution,Includes,herbicides,Chemical
    Agri_Pollution,Agri_Process,Includes,pesticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,insect repellents,Chemical
    Agri_Pollution,Agri_Process,Includes,bug killers,O
    Agricultural contaminants,Agri_Pollution,Includes,pest management chemicals,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,insecticides,Chemical
    farm residue,O,Includes,animal wastes,Agri_Waste
    plant debris,O,Includes,animal wastes,Agri_Waste
    waste material,O,Includes,animal wastes,Agri_Waste
    agricultural waste,O,Includes,animal wastes,Agri_Waste
    biodegradable waste,Agri_Waste,Includes,animal wastes,Agri_Waste
    organic matter,Natural_Resource,Includes,animal wastes,Agri_Waste
    Agri_Residue,Crop,Includes,decomposing botanical substances,O
    Organic debris,Agri_Waste,Includes,rotting vegetation,O
    Agri_Waste,Agri_Process,Includes,deteriorating flora matter,O
    Organic material,Natural_Resource,Includes,decayed plant elements,O
    Agricultural waste,O,Includes,decomposing plant material,O
    organic matter,Natural_Resource,Includes,decaying plant material,O
    irrigation residues,Agri_Waste,Includes,salts,O
    irrigation waste,O,Includes,trace elements,O
    irrigation leftovers,O,Includes,trace metals,O
    farming residues,O,Includes,trace elements,O
    agricultural waste,O,Includes,trace metals,O
    irrigation byproducts,Agri_Waste,Includes,trace elements,O
    irrigation residues,Agri_Waste,Includes,trace metals,O
    irrigation residues,Agri_Waste,Includes,microorganisms,Organism
    Phosphorus,Nutrient,Synonym_Of,P,O
    Nutrient,Nutrient,Synonym_Of,Phosphorus,Nutrient
    Essential element,O,Synonym_Of,Phosphorus,Nutrient
    Mineral,Natural_Resource,Synonym_Of,Phosphorus,Nutrient
    Plant nutrient,Nutrient,Synonym_Of,Phosphorus,Nutrient
    P,O,Synonym_Of,Phosphate,Nutrient
    P,O,Synonym_Of,Phosphorus,Nutrient
    Nitrogenous,O,Helps_In,crop production,O
    Nitrogen element,Nutrient,Helps_In,yield of crops,O
    Nitrogen-rich,Nutrient,Helps_In,agricultural output,O
    Nitrate,Nutrient,Helps_In,crop yield,O
    Nitrogen fertilizer,Nutrient,Helps_In,harvest productivity,O
    nitrogen,Nutrient,Helps_In,crop yield,O
    P,O,Helps_In,crop production,O
    phosphorus,Natural_Resource,Synonym_Of,P,O
    P,O,Synonym_Of,phosphorus,Natural_Resource
    Polluting the soil,O,Caused_By,horticulture,Agri_Process
    Soil contamination,Agri_Pollution,Caused_By,horticultural practices,Agri_Process
    Soil pollution,Agri_Pollution,Caused_By,horticulture,Agri_Process
    Contaminating the land,O,Caused_By,horticultural activities,Agri_Process
    Soil adulteration,Agri_Pollution,Caused_By,horticulture,Agri_Process
    contaminating the soils,Agri_Pollution,Caused_By,horticultural,Agri_Process
    sedimentary deposits,O,Caused_By,horticultural activities,Agri_Process
    sediment,O,Caused_By,horticultural practices,Agri_Process
    sedimentation,O,Caused_By,horticultural impacts,Agri_Process
    sediments,O,Caused_By,horticultural,Agri_Process
    Polluting the farmland,O,Caused_By,industrial activities,O
    Soil contamination,Agri_Pollution,Caused_By,industrial activities,O
    Tainting the soil,O,Caused_By,industrial activities,O
    Contaminating the cropland,O,Caused_By,industrial activities,O
    Pollution of agricultural soils,O,Caused_By,industrial activities,O
    contaminating the soils,Agri_Pollution,Caused_By,industrial activities,O
    Agri_Pollution,Agri_Process,Caused_By,manufacturing operations,O
    Sedimentary deposits,O,Caused_By,industrial processes,O
    Agri_Pollutant,Crop,Caused_By,industrial activities,O
    Sediment,O,Caused_By,industrial operations,O
    Agri_Pollution,Agri_Process,Caused_By,industrial practices,O
    sediments,O,Caused_By,industrial activities,O
    Agri_Pollution,Agri_Process,Includes,natural and synthetic fertilizers,O
    Agricultural pollutants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollutants,Crop,Includes,natural and synthetic fertilizers,O
    Farm pollutants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agricultural pollution,O,Includes,natural and synthetic fertilizers,O
    Agricultural contaminants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollution,Agri_Process,Includes,insecticides,Chemical
    Agricultural pollutants,Agri_Pollution,Includes,pesticides,Chemical
    Farming pollution,O,Includes,pest control substances,Agri_Pollution
    Agri_Contaminants,Chemical,Includes,chemical pesticides,Chemical
    Agricultural pollution,O,Includes,herbicides,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,pesticides,Chemical
    Agri_Pollutants,Crop,Includes,weed killers,O
    Farming pollutants,Agri_Pollution,Includes,herbicides,Chemical
    Agricultural pollution,O,Includes,herbicides,Chemical
    Agri_Contaminants,Chemical,Includes,weed killers,O
    Farm pollutants,Agri_Pollution,Includes,herbicides,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,herbicides,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,insecticides,Chemical
    Gardening,O,Conjunction,Farming,O
    Horticultural,O,Conjunction,Agricultural,O
    Orcharding,Agri_Process,Conjunction,Agricultural,O
    Floricultural,O,Conjunction,Agricultural,O
    Gardening,O,Conjunction,Farming,O
    horticultural,Agri_Process,Conjunction,Agricultural,O
    Soil pollution,Agri_Pollution,Caused_By,horticultural practices,Agri_Process
    Contaminating farmlands,O,Caused_By,horticulture,Agri_Process
    Soil contamination,Agri_Pollution,Caused_By,horticultural activities,Agri_Process
    Polluting the earth,O,Caused_By,gardening,O
    Soil degradation,Agri_Pollution,Caused_By,horticultural methods,Agri_Process
    contaminating the soils,Agri_Pollution,Caused_By,horticultural,Agri_Process
    soil erosion,O,Caused_By,horticultural practices,Agri_Process
    deposits,O,Caused_By,horticultural activities,Agri_Process
    silt,O,Caused_By,horticultural impact,Agri_Process
    particle sediments,O,Caused_By,horticultural methods,Agri_Process
    sediments,O,Caused_By,horticultural,Agri_Process
    Polluting the land,O,Caused_By,Agricultural,O
    Soil contamination,Agri_Pollution,Caused_By,Agricultural,O
    Contaminating the farmland,O,Caused_By,Agricultural,O
    Land pollution,O,Caused_By,Agricultural,O
    Soil degradation,Agri_Pollution,Caused_By,Agricultural,O
    contaminating the soils,Agri_Pollution,Caused_By,Agricultural,O
    sedimentation,O,Caused_By,Farming practices,O
    deposits,O,Caused_By,Agricultural activities,O
    silt,O,Caused_By,Agriculture,Agri_Process
    alluvium,Crop,Caused_By,Farming,O
    settled particles,O,Caused_By,Agricultural practices,O
    sediments,O,Caused_By,Agricultural,O
    Agri_Pollution,Agri_Process,Includes,natural and synthetic fertilizers,O
    Agricultural pollutants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollution,Agri_Process,Includes,natural and chemical fertilizers,O
    Agricultural contaminants,Agri_Pollution,Includes,natural and chemical fertilizers,O
    Agri_Pollution,Agri_Process,Includes,organic and chemical fertilizers,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollutants,Crop,Includes,insecticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,pest control substances,Agri_Pollution
    Agricultural pollution,O,Includes,herbicides,Chemical
    Agri_Pollutants,Crop,Includes,chemical pesticides,Chemical
    Farming contaminants,Agri_Pollution,Includes,pest management products,O
    Agricultural contaminants,Agri_Pollution,Includes,pesticides,Chemical
    Agri_Pollution,Agri_Process,Includes,weed killers,O
    Agricultural pollutants,Agri_Pollution,Includes,herbicides,Chemical
    Farm pollution,O,Includes,weed control chemicals,Chemical
    Agri_Contaminants,Chemical,Includes,herbicides,Chemical
    Agricultural pollution,O,Includes,weed management agents,Person
    Agricultural contaminants,Agri_Pollution,Includes,herbicides,Chemical
    Agri_Pollution,Agri_Process,Includes,pesticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,insect repellents,Chemical
    Agri_Pollution,Agri_Process,Includes,bug killers,O
    Agricultural contaminants,Agri_Pollution,Includes,pest management chemicals,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,insecticides,Chemical
    farm residue,O,Includes,animal wastes,Agri_Waste
    plant debris,O,Includes,animal wastes,Agri_Waste
    waste material,O,Includes,animal wastes,Agri_Waste
    agricultural waste,O,Includes,animal wastes,Agri_Waste
    biodegradable waste,Agri_Waste,Includes,animal wastes,Agri_Waste
    organic matter,Natural_Resource,Includes,animal wastes,Agri_Waste
    Agri_Residue,Crop,Includes,decomposing botanical substances,O
    Organic debris,Agri_Waste,Includes,rotting vegetation,O
    Agri_Waste,Agri_Process,Includes,deteriorating flora matter,O
    Organic material,Natural_Resource,Includes,decayed plant elements,O
    Agricultural waste,O,Includes,decomposing plant material,O
    organic matter,Natural_Resource,Includes,decaying plant material,O
    irrigation residues,Agri_Waste,Includes,salts,O
    irrigation waste,O,Includes,trace elements,O
    irrigation leftovers,O,Includes,trace metals,O
    farming residues,O,Includes,trace elements,O
    agricultural waste,O,Includes,trace metals,O
    irrigation byproducts,Agri_Waste,Includes,trace elements,O
    irrigation residues,Agri_Waste,Includes,trace metals,O
    irrigation residues,Agri_Waste,Includes,microorganisms,Organism
    Phosphorus,Nutrient,Synonym_Of,P,O
    Nutrient,Nutrient,Synonym_Of,Phosphorus,Nutrient
    Essential element,O,Synonym_Of,Phosphorus,Nutrient
    Mineral,Natural_Resource,Synonym_Of,Phosphorus,Nutrient
    Plant nutrient,Nutrient,Synonym_Of,Phosphorus,Nutrient
    P,O,Synonym_Of,Phosphate,Nutrient
    P,O,Synonym_Of,Phosphorus,Nutrient
    Nitrogenous,O,Helps_In,crop production,O
    Nitrogen element,Nutrient,Helps_In,yield of crops,O
    Nitrogen-rich,Nutrient,Helps_In,agricultural output,O
    Nitrate,Nutrient,Helps_In,crop yield,O
    Nitrogen fertilizer,Nutrient,Helps_In,harvest productivity,O
    nitrogen,Nutrient,Helps_In,crop yield,O
    P,O,Helps_In,crop production,O
    phosphorus,Natural_Resource,Synonym_Of,P,O
    P,O,Synonym_Of,phosphorus,Natural_Resource
    Polluting the soil,O,Caused_By,horticulture,Agri_Process
    Soil contamination,Agri_Pollution,Caused_By,horticultural practices,Agri_Process
    Soil pollution,Agri_Pollution,Caused_By,horticulture,Agri_Process
    Contaminating the land,O,Caused_By,horticultural activities,Agri_Process
    Soil adulteration,Agri_Pollution,Caused_By,horticulture,Agri_Process
    contaminating the soils,Agri_Pollution,Caused_By,horticultural,Agri_Process
    sedimentary deposits,O,Caused_By,horticultural activities,Agri_Process
    sediment,O,Caused_By,horticultural practices,Agri_Process
    sedimentation,O,Caused_By,horticultural impacts,Agri_Process
    sediments,O,Caused_By,horticultural,Agri_Process
    Polluting the farmland,O,Caused_By,industrial activities,O
    Soil contamination,Agri_Pollution,Caused_By,industrial activities,O
    Tainting the soil,O,Caused_By,industrial activities,O
    Contaminating the cropland,O,Caused_By,industrial activities,O
    Pollution of agricultural soils,O,Caused_By,industrial activities,O
    contaminating the soils,Agri_Pollution,Caused_By,industrial activities,O
    Agri_Pollution,Agri_Process,Caused_By,manufacturing operations,O
    Sedimentary deposits,O,Caused_By,industrial processes,O
    Agri_Pollutant,Crop,Caused_By,industrial activities,O
    Sediment,O,Caused_By,industrial operations,O
    Agri_Pollution,Agri_Process,Caused_By,industrial practices,O
    sediments,O,Caused_By,industrial activities,O
    Agri_Pollution,Agri_Process,Includes,natural and synthetic fertilizers,O
    Agricultural pollutants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollutants,Crop,Includes,natural and synthetic fertilizers,O
    Farm pollutants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agricultural pollution,O,Includes,natural and synthetic fertilizers,O
    Agricultural contaminants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollution,Agri_Process,Includes,insecticides,Chemical
    Agricultural pollutants,Agri_Pollution,Includes,pesticides,Chemical
    Farming pollution,O,Includes,pest control substances,Agri_Pollution
    Agri_Contaminants,Chemical,Includes,chemical pesticides,Chemical
    Agricultural pollution,O,Includes,herbicides,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,pesticides,Chemical
    Agri_Pollutants,Crop,Includes,weed killers,O
    Farming pollutants,Agri_Pollution,Includes,herbicides,Chemical
    Agricultural pollution,O,Includes,herbicides,Chemical
    Agri_Contaminants,Chemical,Includes,weed killers,O
    Farm pollutants,Agri_Pollution,Includes,herbicides,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,herbicides,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,insecticides,Chemical
    Agri_Pollution,Agri_Process,Includes,organic material,Natural_Resource
    Agricultural pollutants,Agri_Pollution,Includes,organic matter,Natural_Resource
    Agri_Pollution,Agri_Process,Includes,organic matter,Natural_Resource
    Agricultural pollution,O,Includes,organic material,Natural_Resource
    Agricultural contaminants,Agri_Pollution,Includes,organic matter,Natural_Resource
    Gardening,O,Conjunction,Farming,O
    Horticultural,O,Conjunction,Agricultural,O
    Orcharding,Agri_Process,Conjunction,Agricultural,O
    Floricultural,O,Conjunction,Agricultural,O
    Gardening,O,Conjunction,Farming,O
    horticultural,Agri_Process,Conjunction,Agricultural,O
    Soil pollution,Agri_Pollution,Caused_By,horticultural practices,Agri_Process
    Contaminating farmlands,O,Caused_By,horticulture,Agri_Process
    Soil contamination,Agri_Pollution,Caused_By,horticultural activities,Agri_Process
    Polluting the earth,O,Caused_By,gardening,O
    Soil degradation,Agri_Pollution,Caused_By,horticultural methods,Agri_Process
    contaminating the soils,Agri_Pollution,Caused_By,horticultural,Agri_Process
    soil erosion,O,Caused_By,horticultural practices,Agri_Process
    deposits,O,Caused_By,horticultural activities,Agri_Process
    silt,O,Caused_By,horticultural impact,Agri_Process
    particle sediments,O,Caused_By,horticultural methods,Agri_Process
    sediments,O,Caused_By,horticultural,Agri_Process
    Polluting the land,O,Caused_By,Agricultural,O
    Soil contamination,Agri_Pollution,Caused_By,Agricultural,O
    Contaminating the farmland,O,Caused_By,Agricultural,O
    Land pollution,O,Caused_By,Agricultural,O
    Soil degradation,Agri_Pollution,Caused_By,Agricultural,O
    contaminating the soils,Agri_Pollution,Caused_By,Agricultural,O
    sedimentation,O,Caused_By,Farming practices,O
    deposits,O,Caused_By,Agricultural activities,O
    silt,O,Caused_By,Agriculture,Agri_Process
    alluvium,Crop,Caused_By,Farming,O
    settled particles,O,Caused_By,Agricultural practices,O
    sediments,O,Caused_By,Agricultural,O
    Agri_Pollution,Agri_Process,Includes,natural and synthetic fertilizers,O
    Agricultural pollutants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollution,Agri_Process,Includes,natural and chemical fertilizers,O
    Agricultural contaminants,Agri_Pollution,Includes,natural and chemical fertilizers,O
    Agri_Pollution,Agri_Process,Includes,organic and chemical fertilizers,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollutants,Crop,Includes,insecticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,pest control substances,Agri_Pollution
    Agricultural pollution,O,Includes,herbicides,Chemical
    Agri_Pollutants,Crop,Includes,chemical pesticides,Chemical
    Farming contaminants,Agri_Pollution,Includes,pest management products,O
    Agricultural contaminants,Agri_Pollution,Includes,pesticides,Chemical
    Agri_Pollution,Agri_Process,Includes,weed killers,O
    Agricultural pollutants,Agri_Pollution,Includes,herbicides,Chemical
    Farm pollution,O,Includes,weed control chemicals,Chemical
    Agri_Contaminants,Chemical,Includes,herbicides,Chemical
    Agricultural pollution,O,Includes,weed management agents,Person
    Agricultural contaminants,Agri_Pollution,Includes,herbicides,Chemical
    Agri_Pollution,Agri_Process,Includes,pesticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,insect repellents,Chemical
    Agri_Pollution,Agri_Process,Includes,bug killers,O
    Agricultural contaminants,Agri_Pollution,Includes,pest management chemicals,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,insecticides,Chemical
    farm residue,O,Includes,animal wastes,Agri_Waste
    plant debris,O,Includes,animal wastes,Agri_Waste
    waste material,O,Includes,animal wastes,Agri_Waste
    agricultural waste,O,Includes,animal wastes,Agri_Waste
    biodegradable waste,Agri_Waste,Includes,animal wastes,Agri_Waste
    organic matter,Natural_Resource,Includes,animal wastes,Agri_Waste
    Agri_Residue,Crop,Includes,decomposing botanical substances,O
    Organic debris,Agri_Waste,Includes,rotting vegetation,O
    Agri_Waste,Agri_Process,Includes,deteriorating flora matter,O
    Organic material,Natural_Resource,Includes,decayed plant elements,O
    Agricultural waste,O,Includes,decomposing plant material,O
    organic matter,Natural_Resource,Includes,decaying plant material,O
    irrigation residues,Agri_Waste,Includes,salts,O
    irrigation waste,O,Includes,trace elements,O
    irrigation leftovers,O,Includes,trace metals,O
    farming residues,O,Includes,trace elements,O
    agricultural waste,O,Includes,trace metals,O
    irrigation byproducts,Agri_Waste,Includes,trace elements,O
    irrigation residues,Agri_Waste,Includes,trace metals,O
    irrigation residues,Agri_Waste,Includes,microorganisms,Organism
    Phosphorus,Nutrient,Synonym_Of,P,O
    Nutrient,Nutrient,Synonym_Of,Phosphorus,Nutrient
    Essential element,O,Synonym_Of,Phosphorus,Nutrient
    Mineral,Natural_Resource,Synonym_Of,Phosphorus,Nutrient
    Plant nutrient,Nutrient,Synonym_Of,Phosphorus,Nutrient
    P,O,Synonym_Of,Phosphate,Nutrient
    P,O,Synonym_Of,Phosphorus,Nutrient
    Nitrogenous,O,Helps_In,crop production,O
    Nitrogen element,Nutrient,Helps_In,yield of crops,O
    Nitrogen-rich,Nutrient,Helps_In,agricultural output,O
    Nitrate,Nutrient,Helps_In,crop yield,O
    Nitrogen fertilizer,Nutrient,Helps_In,harvest productivity,O
    nitrogen,Nutrient,Helps_In,crop yield,O
    P,O,Helps_In,crop production,O
    phosphorus,Natural_Resource,Synonym_Of,P,O
    P,O,Synonym_Of,phosphorus,Natural_Resource
    Polluting the soil,O,Caused_By,horticulture,Agri_Process
    Soil contamination,Agri_Pollution,Caused_By,horticultural practices,Agri_Process
    Soil pollution,Agri_Pollution,Caused_By,horticulture,Agri_Process
    Contaminating the land,O,Caused_By,horticultural activities,Agri_Process
    Soil adulteration,Agri_Pollution,Caused_By,horticulture,Agri_Process
    contaminating the soils,Agri_Pollution,Caused_By,horticultural,Agri_Process
    sedimentary deposits,O,Caused_By,horticultural activities,Agri_Process
    sediment,O,Caused_By,horticultural practices,Agri_Process
    sedimentation,O,Caused_By,horticultural impacts,Agri_Process
    sediments,O,Caused_By,horticultural,Agri_Process
    Polluting the farmland,O,Caused_By,industrial activities,O
    Soil contamination,Agri_Pollution,Caused_By,industrial activities,O
    Tainting the soil,O,Caused_By,industrial activities,O
    Contaminating the cropland,O,Caused_By,industrial activities,O
    Pollution of agricultural soils,O,Caused_By,industrial activities,O
    contaminating the soils,Agri_Pollution,Caused_By,industrial activities,O
    Agri_Pollution,Agri_Process,Caused_By,manufacturing operations,O
    Sedimentary deposits,O,Caused_By,industrial processes,O
    Agri_Pollutant,Crop,Caused_By,industrial activities,O
    Sediment,O,Caused_By,industrial operations,O
    Agri_Pollution,Agri_Process,Caused_By,industrial practices,O
    sediments,O,Caused_By,industrial activities,O
    Agri_Pollution,Agri_Process,Includes,natural and synthetic fertilizers,O
    Agricultural pollutants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollutants,Crop,Includes,natural and synthetic fertilizers,O
    Farm pollutants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agricultural pollution,O,Includes,natural and synthetic fertilizers,O
    Agricultural contaminants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollution,Agri_Process,Includes,insecticides,Chemical
    Agricultural pollutants,Agri_Pollution,Includes,pesticides,Chemical
    Farming pollution,O,Includes,pest control substances,Agri_Pollution
    Agri_Contaminants,Chemical,Includes,chemical pesticides,Chemical
    Agricultural pollution,O,Includes,herbicides,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,pesticides,Chemical
    Agri_Pollutants,Crop,Includes,weed killers,O
    Farming pollutants,Agri_Pollution,Includes,herbicides,Chemical
    Agricultural pollution,O,Includes,herbicides,Chemical
    Agri_Contaminants,Chemical,Includes,weed killers,O
    Farm pollutants,Agri_Pollution,Includes,herbicides,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,herbicides,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,insecticides,Chemical
    Agri_Pollution,Agri_Process,Includes,organic material,Natural_Resource
    Agricultural pollutants,Agri_Pollution,Includes,organic matter,Natural_Resource
    Agri_Pollution,Agri_Process,Includes,organic matter,Natural_Resource
    Agricultural pollution,O,Includes,organic material,Natural_Resource
    Agricultural contaminants,Agri_Pollution,Includes,organic matter,Natural_Resource
    Agri_Pollutants,Crop,Includes,irrigation remains,O
    Agricultural pollutants,Agri_Pollution,Includes,irrigation remnants,O
    Agri_Pollutants,Crop,Includes,irrigation leftovers,O
    Agricultural contaminants,Agri_Pollution,Includes,irrigation residues,Agri_Waste
    Agri_Pollution,Agri_Process,Includes,irrigation waste,O
    Agricultural contaminants,Agri_Pollution,Includes,irrigation residues,Agri_Waste
    Gardening,O,Conjunction,Farming,O
    Horticultural,O,Conjunction,Agricultural,O
    Orcharding,Agri_Process,Conjunction,Agricultural,O
    Floricultural,O,Conjunction,Agricultural,O
    Gardening,O,Conjunction,Farming,O
    horticultural,Agri_Process,Conjunction,Agricultural,O
    Soil pollution,Agri_Pollution,Caused_By,horticultural practices,Agri_Process
    Contaminating farmlands,O,Caused_By,horticulture,Agri_Process
    Soil contamination,Agri_Pollution,Caused_By,horticultural activities,Agri_Process
    Polluting the earth,O,Caused_By,gardening,O
    Soil degradation,Agri_Pollution,Caused_By,horticultural methods,Agri_Process
    contaminating the soils,Agri_Pollution,Caused_By,horticultural,Agri_Process
    soil erosion,O,Caused_By,horticultural practices,Agri_Process
    deposits,O,Caused_By,horticultural activities,Agri_Process
    silt,O,Caused_By,horticultural impact,Agri_Process
    particle sediments,O,Caused_By,horticultural methods,Agri_Process
    sediments,O,Caused_By,horticultural,Agri_Process
    Polluting the land,O,Caused_By,Agricultural,O
    Soil contamination,Agri_Pollution,Caused_By,Agricultural,O
    Contaminating the farmland,O,Caused_By,Agricultural,O
    Land pollution,O,Caused_By,Agricultural,O
    Soil degradation,Agri_Pollution,Caused_By,Agricultural,O
    contaminating the soils,Agri_Pollution,Caused_By,Agricultural,O
    sedimentation,O,Caused_By,Farming practices,O
    deposits,O,Caused_By,Agricultural activities,O
    silt,O,Caused_By,Agriculture,Agri_Process
    alluvium,Crop,Caused_By,Farming,O
    settled particles,O,Caused_By,Agricultural practices,O
    sediments,O,Caused_By,Agricultural,O
    Agri_Pollution,Agri_Process,Includes,natural and synthetic fertilizers,O
    Agricultural pollutants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollution,Agri_Process,Includes,natural and chemical fertilizers,O
    Agricultural contaminants,Agri_Pollution,Includes,natural and chemical fertilizers,O
    Agri_Pollution,Agri_Process,Includes,organic and chemical fertilizers,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollutants,Crop,Includes,insecticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,pest control substances,Agri_Pollution
    Agricultural pollution,O,Includes,herbicides,Chemical
    Agri_Pollutants,Crop,Includes,chemical pesticides,Chemical
    Farming contaminants,Agri_Pollution,Includes,pest management products,O
    Agricultural contaminants,Agri_Pollution,Includes,pesticides,Chemical
    Agri_Pollution,Agri_Process,Includes,weed killers,O
    Agricultural pollutants,Agri_Pollution,Includes,herbicides,Chemical
    Farm pollution,O,Includes,weed control chemicals,Chemical
    Agri_Contaminants,Chemical,Includes,herbicides,Chemical
    Agricultural pollution,O,Includes,weed management agents,Person
    Agricultural contaminants,Agri_Pollution,Includes,herbicides,Chemical
    Agri_Pollution,Agri_Process,Includes,pesticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,insect repellents,Chemical
    Agri_Pollution,Agri_Process,Includes,bug killers,O
    Agricultural contaminants,Agri_Pollution,Includes,pest management chemicals,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,insecticides,Chemical
    farm residue,O,Includes,animal wastes,Agri_Waste
    plant debris,O,Includes,animal wastes,Agri_Waste
    waste material,O,Includes,animal wastes,Agri_Waste
    agricultural waste,O,Includes,animal wastes,Agri_Waste
    biodegradable waste,Agri_Waste,Includes,animal wastes,Agri_Waste
    organic matter,Natural_Resource,Includes,animal wastes,Agri_Waste
    Agri_Residue,Crop,Includes,decomposing botanical substances,O
    Organic debris,Agri_Waste,Includes,rotting vegetation,O
    Agri_Waste,Agri_Process,Includes,deteriorating flora matter,O
    Organic material,Natural_Resource,Includes,decayed plant elements,O
    Agricultural waste,O,Includes,decomposing plant material,O
    organic matter,Natural_Resource,Includes,decaying plant material,O
    irrigation residues,Agri_Waste,Includes,salts,O
    irrigation waste,O,Includes,trace elements,O
    irrigation leftovers,O,Includes,trace metals,O
    farming residues,O,Includes,trace elements,O
    agricultural waste,O,Includes,trace metals,O
    irrigation byproducts,Agri_Waste,Includes,trace elements,O
    irrigation residues,Agri_Waste,Includes,trace metals,O
    irrigation residues,Agri_Waste,Includes,microorganisms,Organism
    Phosphorus,Nutrient,Synonym_Of,P,O
    Nutrient,Nutrient,Synonym_Of,Phosphorus,Nutrient
    Essential element,O,Synonym_Of,Phosphorus,Nutrient
    Mineral,Natural_Resource,Synonym_Of,Phosphorus,Nutrient
    Plant nutrient,Nutrient,Synonym_Of,Phosphorus,Nutrient
    P,O,Synonym_Of,Phosphate,Nutrient
    P,O,Synonym_Of,Phosphorus,Nutrient
    Nitrogenous,O,Helps_In,crop production,O
    Nitrogen element,Nutrient,Helps_In,yield of crops,O
    Nitrogen-rich,Nutrient,Helps_In,agricultural output,O
    Nitrate,Nutrient,Helps_In,crop yield,O
    Nitrogen fertilizer,Nutrient,Helps_In,harvest productivity,O
    nitrogen,Nutrient,Helps_In,crop yield,O
    P,O,Helps_In,crop production,O
    phosphorus,Natural_Resource,Synonym_Of,P,O
    P,O,Synonym_Of,phosphorus,Natural_Resource
    Polluting the soil,O,Caused_By,horticulture,Agri_Process
    Soil contamination,Agri_Pollution,Caused_By,horticultural practices,Agri_Process
    Soil pollution,Agri_Pollution,Caused_By,horticulture,Agri_Process
    Contaminating the land,O,Caused_By,horticultural activities,Agri_Process
    Soil adulteration,Agri_Pollution,Caused_By,horticulture,Agri_Process
    contaminating the soils,Agri_Pollution,Caused_By,horticultural,Agri_Process
    sedimentary deposits,O,Caused_By,horticultural activities,Agri_Process
    sediment,O,Caused_By,horticultural practices,Agri_Process
    sedimentation,O,Caused_By,horticultural impacts,Agri_Process
    sediments,O,Caused_By,horticultural,Agri_Process
    Polluting the farmland,O,Caused_By,industrial activities,O
    Soil contamination,Agri_Pollution,Caused_By,industrial activities,O
    Tainting the soil,O,Caused_By,industrial activities,O
    Contaminating the cropland,O,Caused_By,industrial activities,O
    Pollution of agricultural soils,O,Caused_By,industrial activities,O
    contaminating the soils,Agri_Pollution,Caused_By,industrial activities,O
    Agri_Pollution,Agri_Process,Caused_By,manufacturing operations,O
    Sedimentary deposits,O,Caused_By,industrial processes,O
    Agri_Pollutant,Crop,Caused_By,industrial activities,O
    Sediment,O,Caused_By,industrial operations,O
    Agri_Pollution,Agri_Process,Caused_By,industrial practices,O
    sediments,O,Caused_By,industrial activities,O
    Agri_Pollution,Agri_Process,Includes,natural and synthetic fertilizers,O
    Agricultural pollutants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollutants,Crop,Includes,natural and synthetic fertilizers,O
    Farm pollutants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agricultural pollution,O,Includes,natural and synthetic fertilizers,O
    Agricultural contaminants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollution,Agri_Process,Includes,insecticides,Chemical
    Agricultural pollutants,Agri_Pollution,Includes,pesticides,Chemical
    Farming pollution,O,Includes,pest control substances,Agri_Pollution
    Agri_Contaminants,Chemical,Includes,chemical pesticides,Chemical
    Agricultural pollution,O,Includes,herbicides,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,pesticides,Chemical
    Agri_Pollutants,Crop,Includes,weed killers,O
    Farming pollutants,Agri_Pollution,Includes,herbicides,Chemical
    Agricultural pollution,O,Includes,herbicides,Chemical
    Agri_Contaminants,Chemical,Includes,weed killers,O
    Farm pollutants,Agri_Pollution,Includes,herbicides,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,herbicides,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,insecticides,Chemical
    Agri_Pollution,Agri_Process,Includes,organic material,Natural_Resource
    Agricultural pollutants,Agri_Pollution,Includes,organic matter,Natural_Resource
    Agri_Pollution,Agri_Process,Includes,organic matter,Natural_Resource
    Agricultural pollution,O,Includes,organic material,Natural_Resource
    Agricultural contaminants,Agri_Pollution,Includes,organic matter,Natural_Resource
    Agri_Pollutants,Crop,Includes,irrigation remains,O
    Agricultural pollutants,Agri_Pollution,Includes,irrigation remnants,O
    Agri_Pollutants,Crop,Includes,irrigation leftovers,O
    Agricultural contaminants,Agri_Pollution,Includes,irrigation residues,Agri_Waste
    Agri_Pollution,Agri_Process,Includes,irrigation waste,O
    Agricultural contaminants,Agri_Pollution,Includes,irrigation residues,Agri_Waste
    Agri_Waste,Agri_Process,Includes,animal byproducts,Agri_Waste
    Organic materials,Natural_Resource,Includes,animal wastes,Agri_Waste
    Agricultural waste,O,Includes,animal excreta,O
    Farm waste,O,Includes,animal residues,Agri_Waste
    Bio waste,O,Includes,animal byproducts,Agri_Waste
    organic matter,Natural_Resource,Includes,animal wastes,Agri_Waste
    Gardening,O,Conjunction,Farming,O
    Horticultural,O,Conjunction,Agricultural,O
    Orcharding,Agri_Process,Conjunction,Agricultural,O
    Floricultural,O,Conjunction,Agricultural,O
    Gardening,O,Conjunction,Farming,O
    horticultural,Agri_Process,Conjunction,Agricultural,O
    Soil pollution,Agri_Pollution,Caused_By,horticultural practices,Agri_Process
    Contaminating farmlands,O,Caused_By,horticulture,Agri_Process
    Soil contamination,Agri_Pollution,Caused_By,horticultural activities,Agri_Process
    Polluting the earth,O,Caused_By,gardening,O
    Soil degradation,Agri_Pollution,Caused_By,horticultural methods,Agri_Process
    contaminating the soils,Agri_Pollution,Caused_By,horticultural,Agri_Process
    soil erosion,O,Caused_By,horticultural practices,Agri_Process
    deposits,O,Caused_By,horticultural activities,Agri_Process
    silt,O,Caused_By,horticultural impact,Agri_Process
    particle sediments,O,Caused_By,horticultural methods,Agri_Process
    sediments,O,Caused_By,horticultural,Agri_Process
    Polluting the land,O,Caused_By,Agricultural,O
    Soil contamination,Agri_Pollution,Caused_By,Agricultural,O
    Contaminating the farmland,O,Caused_By,Agricultural,O
    Land pollution,O,Caused_By,Agricultural,O
    Soil degradation,Agri_Pollution,Caused_By,Agricultural,O
    contaminating the soils,Agri_Pollution,Caused_By,Agricultural,O
    sedimentation,O,Caused_By,Farming practices,O
    deposits,O,Caused_By,Agricultural activities,O
    silt,O,Caused_By,Agriculture,Agri_Process
    alluvium,Crop,Caused_By,Farming,O
    settled particles,O,Caused_By,Agricultural practices,O
    sediments,O,Caused_By,Agricultural,O
    Agri_Pollution,Agri_Process,Includes,natural and synthetic fertilizers,O
    Agricultural pollutants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollution,Agri_Process,Includes,natural and chemical fertilizers,O
    Agricultural contaminants,Agri_Pollution,Includes,natural and chemical fertilizers,O
    Agri_Pollution,Agri_Process,Includes,organic and chemical fertilizers,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollutants,Crop,Includes,insecticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,pest control substances,Agri_Pollution
    Agricultural pollution,O,Includes,herbicides,Chemical
    Agri_Pollutants,Crop,Includes,chemical pesticides,Chemical
    Farming contaminants,Agri_Pollution,Includes,pest management products,O
    Agricultural contaminants,Agri_Pollution,Includes,pesticides,Chemical
    Agri_Pollution,Agri_Process,Includes,weed killers,O
    Agricultural pollutants,Agri_Pollution,Includes,herbicides,Chemical
    Farm pollution,O,Includes,weed control chemicals,Chemical
    Agri_Contaminants,Chemical,Includes,herbicides,Chemical
    Agricultural pollution,O,Includes,weed management agents,Person
    Agricultural contaminants,Agri_Pollution,Includes,herbicides,Chemical
    Agri_Pollution,Agri_Process,Includes,pesticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,insect repellents,Chemical
    Agri_Pollution,Agri_Process,Includes,bug killers,O
    Agricultural contaminants,Agri_Pollution,Includes,pest management chemicals,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,insecticides,Chemical
    farm residue,O,Includes,animal wastes,Agri_Waste
    plant debris,O,Includes,animal wastes,Agri_Waste
    waste material,O,Includes,animal wastes,Agri_Waste
    agricultural waste,O,Includes,animal wastes,Agri_Waste
    biodegradable waste,Agri_Waste,Includes,animal wastes,Agri_Waste
    organic matter,Natural_Resource,Includes,animal wastes,Agri_Waste
    Agri_Residue,Crop,Includes,decomposing botanical substances,O
    Organic debris,Agri_Waste,Includes,rotting vegetation,O
    Agri_Waste,Agri_Process,Includes,deteriorating flora matter,O
    Organic material,Natural_Resource,Includes,decayed plant elements,O
    Agricultural waste,O,Includes,decomposing plant material,O
    organic matter,Natural_Resource,Includes,decaying plant material,O
    irrigation residues,Agri_Waste,Includes,salts,O
    irrigation waste,O,Includes,trace elements,O
    irrigation leftovers,O,Includes,trace metals,O
    farming residues,O,Includes,trace elements,O
    agricultural waste,O,Includes,trace metals,O
    irrigation byproducts,Agri_Waste,Includes,trace elements,O
    irrigation residues,Agri_Waste,Includes,trace metals,O
    irrigation residues,Agri_Waste,Includes,microorganisms,Organism
    Phosphorus,Nutrient,Synonym_Of,P,O
    Nutrient,Nutrient,Synonym_Of,Phosphorus,Nutrient
    Essential element,O,Synonym_Of,Phosphorus,Nutrient
    Mineral,Natural_Resource,Synonym_Of,Phosphorus,Nutrient
    Plant nutrient,Nutrient,Synonym_Of,Phosphorus,Nutrient
    P,O,Synonym_Of,Phosphate,Nutrient
    P,O,Synonym_Of,Phosphorus,Nutrient
    Nitrogenous,O,Helps_In,crop production,O
    Nitrogen element,Nutrient,Helps_In,yield of crops,O
    Nitrogen-rich,Nutrient,Helps_In,agricultural output,O
    Nitrate,Nutrient,Helps_In,crop yield,O
    Nitrogen fertilizer,Nutrient,Helps_In,harvest productivity,O
    nitrogen,Nutrient,Helps_In,crop yield,O
    P,O,Helps_In,crop production,O
    phosphorus,Natural_Resource,Synonym_Of,P,O
    P,O,Synonym_Of,phosphorus,Natural_Resource
    Polluting the soil,O,Caused_By,horticulture,Agri_Process
    Soil contamination,Agri_Pollution,Caused_By,horticultural practices,Agri_Process
    Soil pollution,Agri_Pollution,Caused_By,horticulture,Agri_Process
    Contaminating the land,O,Caused_By,horticultural activities,Agri_Process
    Soil adulteration,Agri_Pollution,Caused_By,horticulture,Agri_Process
    contaminating the soils,Agri_Pollution,Caused_By,horticultural,Agri_Process
    sedimentary deposits,O,Caused_By,horticultural activities,Agri_Process
    sediment,O,Caused_By,horticultural practices,Agri_Process
    sedimentation,O,Caused_By,horticultural impacts,Agri_Process
    sediments,O,Caused_By,horticultural,Agri_Process
    Polluting the farmland,O,Caused_By,industrial activities,O
    Soil contamination,Agri_Pollution,Caused_By,industrial activities,O
    Tainting the soil,O,Caused_By,industrial activities,O
    Contaminating the cropland,O,Caused_By,industrial activities,O
    Pollution of agricultural soils,O,Caused_By,industrial activities,O
    contaminating the soils,Agri_Pollution,Caused_By,industrial activities,O
    Agri_Pollution,Agri_Process,Caused_By,manufacturing operations,O
    Sedimentary deposits,O,Caused_By,industrial processes,O
    Agri_Pollutant,Crop,Caused_By,industrial activities,O
    Sediment,O,Caused_By,industrial operations,O
    Agri_Pollution,Agri_Process,Caused_By,industrial practices,O
    sediments,O,Caused_By,industrial activities,O
    Agri_Pollution,Agri_Process,Includes,natural and synthetic fertilizers,O
    Agricultural pollutants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollutants,Crop,Includes,natural and synthetic fertilizers,O
    Farm pollutants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agricultural pollution,O,Includes,natural and synthetic fertilizers,O
    Agricultural contaminants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollution,Agri_Process,Includes,insecticides,Chemical
    Agricultural pollutants,Agri_Pollution,Includes,pesticides,Chemical
    Farming pollution,O,Includes,pest control substances,Agri_Pollution
    Agri_Contaminants,Chemical,Includes,chemical pesticides,Chemical
    Agricultural pollution,O,Includes,herbicides,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,pesticides,Chemical
    Agri_Pollutants,Crop,Includes,weed killers,O
    Farming pollutants,Agri_Pollution,Includes,herbicides,Chemical
    Agricultural pollution,O,Includes,herbicides,Chemical
    Agri_Contaminants,Chemical,Includes,weed killers,O
    Farm pollutants,Agri_Pollution,Includes,herbicides,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,herbicides,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,insecticides,Chemical
    Agri_Pollution,Agri_Process,Includes,organic material,Natural_Resource
    Agricultural pollutants,Agri_Pollution,Includes,organic matter,Natural_Resource
    Agri_Pollution,Agri_Process,Includes,organic matter,Natural_Resource
    Agricultural pollution,O,Includes,organic material,Natural_Resource
    Agricultural contaminants,Agri_Pollution,Includes,organic matter,Natural_Resource
    Agri_Pollutants,Crop,Includes,irrigation remains,O
    Agricultural pollutants,Agri_Pollution,Includes,irrigation remnants,O
    Agri_Pollutants,Crop,Includes,irrigation leftovers,O
    Agricultural contaminants,Agri_Pollution,Includes,irrigation residues,Agri_Waste
    Agri_Pollution,Agri_Process,Includes,irrigation waste,O
    Agricultural contaminants,Agri_Pollution,Includes,irrigation residues,Agri_Waste
    Agri_Waste,Agri_Process,Includes,animal byproducts,Agri_Waste
    Organic materials,Natural_Resource,Includes,animal wastes,Agri_Waste
    Agricultural waste,O,Includes,animal excreta,O
    Farm waste,O,Includes,animal residues,Agri_Waste
    Bio waste,O,Includes,animal byproducts,Agri_Waste
    organic matter,Natural_Resource,Includes,animal wastes,Agri_Waste
    Agri_Residue,Crop,Includes,decaying vegetation,O
    Organic waste,Natural_Resource,Includes,deteriorating plant material,O
    Biodegradable waste,Chemical,Includes,rotting botanical matter,O
    Agri_Waste,Agri_Process,Includes,decomposing plant material,O
    Organic material,Natural_Resource,Includes,decayed plant substance,Chemical
    organic matter,Natural_Resource,Includes,decaying plant material,O
    Gardening,O,Conjunction,Farming,O
    Horticultural,O,Conjunction,Agricultural,O
    Orcharding,Agri_Process,Conjunction,Agricultural,O
    Floricultural,O,Conjunction,Agricultural,O
    Gardening,O,Conjunction,Farming,O
    horticultural,Agri_Process,Conjunction,Agricultural,O
    Soil pollution,Agri_Pollution,Caused_By,horticultural practices,Agri_Process
    Contaminating farmlands,O,Caused_By,horticulture,Agri_Process
    Soil contamination,Agri_Pollution,Caused_By,horticultural activities,Agri_Process
    Polluting the earth,O,Caused_By,gardening,O
    Soil degradation,Agri_Pollution,Caused_By,horticultural methods,Agri_Process
    contaminating the soils,Agri_Pollution,Caused_By,horticultural,Agri_Process
    soil erosion,O,Caused_By,horticultural practices,Agri_Process
    deposits,O,Caused_By,horticultural activities,Agri_Process
    silt,O,Caused_By,horticultural impact,Agri_Process
    particle sediments,O,Caused_By,horticultural methods,Agri_Process
    sediments,O,Caused_By,horticultural,Agri_Process
    Polluting the land,O,Caused_By,Agricultural,O
    Soil contamination,Agri_Pollution,Caused_By,Agricultural,O
    Contaminating the farmland,O,Caused_By,Agricultural,O
    Land pollution,O,Caused_By,Agricultural,O
    Soil degradation,Agri_Pollution,Caused_By,Agricultural,O
    contaminating the soils,Agri_Pollution,Caused_By,Agricultural,O
    sedimentation,O,Caused_By,Farming practices,O
    deposits,O,Caused_By,Agricultural activities,O
    silt,O,Caused_By,Agriculture,Agri_Process
    alluvium,Crop,Caused_By,Farming,O
    settled particles,O,Caused_By,Agricultural practices,O
    sediments,O,Caused_By,Agricultural,O
    Agri_Pollution,Agri_Process,Includes,natural and synthetic fertilizers,O
    Agricultural pollutants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollution,Agri_Process,Includes,natural and chemical fertilizers,O
    Agricultural contaminants,Agri_Pollution,Includes,natural and chemical fertilizers,O
    Agri_Pollution,Agri_Process,Includes,organic and chemical fertilizers,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollutants,Crop,Includes,insecticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,pest control substances,Agri_Pollution
    Agricultural pollution,O,Includes,herbicides,Chemical
    Agri_Pollutants,Crop,Includes,chemical pesticides,Chemical
    Farming contaminants,Agri_Pollution,Includes,pest management products,O
    Agricultural contaminants,Agri_Pollution,Includes,pesticides,Chemical
    Agri_Pollution,Agri_Process,Includes,weed killers,O
    Agricultural pollutants,Agri_Pollution,Includes,herbicides,Chemical
    Farm pollution,O,Includes,weed control chemicals,Chemical
    Agri_Contaminants,Chemical,Includes,herbicides,Chemical
    Agricultural pollution,O,Includes,weed management agents,Person
    Agricultural contaminants,Agri_Pollution,Includes,herbicides,Chemical
    Agri_Pollution,Agri_Process,Includes,pesticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,insect repellents,Chemical
    Agri_Pollution,Agri_Process,Includes,bug killers,O
    Agricultural contaminants,Agri_Pollution,Includes,pest management chemicals,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,insecticides,Chemical
    farm residue,O,Includes,animal wastes,Agri_Waste
    plant debris,O,Includes,animal wastes,Agri_Waste
    waste material,O,Includes,animal wastes,Agri_Waste
    agricultural waste,O,Includes,animal wastes,Agri_Waste
    biodegradable waste,Agri_Waste,Includes,animal wastes,Agri_Waste
    organic matter,Natural_Resource,Includes,animal wastes,Agri_Waste
    Agri_Residue,Crop,Includes,decomposing botanical substances,O
    Organic debris,Agri_Waste,Includes,rotting vegetation,O
    Agri_Waste,Agri_Process,Includes,deteriorating flora matter,O
    Organic material,Natural_Resource,Includes,decayed plant elements,O
    Agricultural waste,O,Includes,decomposing plant material,O
    organic matter,Natural_Resource,Includes,decaying plant material,O
    irrigation residues,Agri_Waste,Includes,salts,O
    irrigation waste,O,Includes,trace elements,O
    irrigation leftovers,O,Includes,trace metals,O
    farming residues,O,Includes,trace elements,O
    agricultural waste,O,Includes,trace metals,O
    irrigation byproducts,Agri_Waste,Includes,trace elements,O
    irrigation residues,Agri_Waste,Includes,trace metals,O
    irrigation residues,Agri_Waste,Includes,microorganisms,Organism
    Phosphorus,Nutrient,Synonym_Of,P,O
    Nutrient,Nutrient,Synonym_Of,Phosphorus,Nutrient
    Essential element,O,Synonym_Of,Phosphorus,Nutrient
    Mineral,Natural_Resource,Synonym_Of,Phosphorus,Nutrient
    Plant nutrient,Nutrient,Synonym_Of,Phosphorus,Nutrient
    P,O,Synonym_Of,Phosphate,Nutrient
    P,O,Synonym_Of,Phosphorus,Nutrient
    Nitrogenous,O,Helps_In,crop production,O
    Nitrogen element,Nutrient,Helps_In,yield of crops,O
    Nitrogen-rich,Nutrient,Helps_In,agricultural output,O
    Nitrate,Nutrient,Helps_In,crop yield,O
    Nitrogen fertilizer,Nutrient,Helps_In,harvest productivity,O
    nitrogen,Nutrient,Helps_In,crop yield,O
    P,O,Helps_In,crop production,O
    phosphorus,Natural_Resource,Synonym_Of,P,O
    P,O,Synonym_Of,phosphorus,Natural_Resource
    Polluting the soil,O,Caused_By,horticulture,Agri_Process
    Soil contamination,Agri_Pollution,Caused_By,horticultural practices,Agri_Process
    Soil pollution,Agri_Pollution,Caused_By,horticulture,Agri_Process
    Contaminating the land,O,Caused_By,horticultural activities,Agri_Process
    Soil adulteration,Agri_Pollution,Caused_By,horticulture,Agri_Process
    contaminating the soils,Agri_Pollution,Caused_By,horticultural,Agri_Process
    sedimentary deposits,O,Caused_By,horticultural activities,Agri_Process
    sediment,O,Caused_By,horticultural practices,Agri_Process
    sedimentation,O,Caused_By,horticultural impacts,Agri_Process
    sediments,O,Caused_By,horticultural,Agri_Process
    Polluting the farmland,O,Caused_By,industrial activities,O
    Soil contamination,Agri_Pollution,Caused_By,industrial activities,O
    Tainting the soil,O,Caused_By,industrial activities,O
    Contaminating the cropland,O,Caused_By,industrial activities,O
    Pollution of agricultural soils,O,Caused_By,industrial activities,O
    contaminating the soils,Agri_Pollution,Caused_By,industrial activities,O
    Agri_Pollution,Agri_Process,Caused_By,manufacturing operations,O
    Sedimentary deposits,O,Caused_By,industrial processes,O
    Agri_Pollutant,Crop,Caused_By,industrial activities,O
    Sediment,O,Caused_By,industrial operations,O
    Agri_Pollution,Agri_Process,Caused_By,industrial practices,O
    sediments,O,Caused_By,industrial activities,O
    Agri_Pollution,Agri_Process,Includes,natural and synthetic fertilizers,O
    Agricultural pollutants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollutants,Crop,Includes,natural and synthetic fertilizers,O
    Farm pollutants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agricultural pollution,O,Includes,natural and synthetic fertilizers,O
    Agricultural contaminants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollution,Agri_Process,Includes,insecticides,Chemical
    Agricultural pollutants,Agri_Pollution,Includes,pesticides,Chemical
    Farming pollution,O,Includes,pest control substances,Agri_Pollution
    Agri_Contaminants,Chemical,Includes,chemical pesticides,Chemical
    Agricultural pollution,O,Includes,herbicides,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,pesticides,Chemical
    Agri_Pollutants,Crop,Includes,weed killers,O
    Farming pollutants,Agri_Pollution,Includes,herbicides,Chemical
    Agricultural pollution,O,Includes,herbicides,Chemical
    Agri_Contaminants,Chemical,Includes,weed killers,O
    Farm pollutants,Agri_Pollution,Includes,herbicides,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,herbicides,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,insecticides,Chemical
    Agri_Pollution,Agri_Process,Includes,organic material,Natural_Resource
    Agricultural pollutants,Agri_Pollution,Includes,organic matter,Natural_Resource
    Agri_Pollution,Agri_Process,Includes,organic matter,Natural_Resource
    Agricultural pollution,O,Includes,organic material,Natural_Resource
    Agricultural contaminants,Agri_Pollution,Includes,organic matter,Natural_Resource
    Agri_Pollutants,Crop,Includes,irrigation remains,O
    Agricultural pollutants,Agri_Pollution,Includes,irrigation remnants,O
    Agri_Pollutants,Crop,Includes,irrigation leftovers,O
    Agricultural contaminants,Agri_Pollution,Includes,irrigation residues,Agri_Waste
    Agri_Pollution,Agri_Process,Includes,irrigation waste,O
    Agricultural contaminants,Agri_Pollution,Includes,irrigation residues,Agri_Waste
    Agri_Waste,Agri_Process,Includes,animal byproducts,Agri_Waste
    Organic materials,Natural_Resource,Includes,animal wastes,Agri_Waste
    Agricultural waste,O,Includes,animal excreta,O
    Farm waste,O,Includes,animal residues,Agri_Waste
    Bio waste,O,Includes,animal byproducts,Agri_Waste
    organic matter,Natural_Resource,Includes,animal wastes,Agri_Waste
    Agri_Residue,Crop,Includes,decaying vegetation,O
    Organic waste,Natural_Resource,Includes,deteriorating plant material,O
    Biodegradable waste,Chemical,Includes,rotting botanical matter,O
    Agri_Waste,Agri_Process,Includes,decomposing plant material,O
    Organic material,Natural_Resource,Includes,decayed plant substance,Chemical
    organic matter,Natural_Resource,Includes,decaying plant material,O
    irrigation residues,Agri_Waste,Includes,salts,O
    Gardening,O,Conjunction,Farming,O
    Horticultural,O,Conjunction,Agricultural,O
    Orcharding,Agri_Process,Conjunction,Agricultural,O
    Floricultural,O,Conjunction,Agricultural,O
    Gardening,O,Conjunction,Farming,O
    horticultural,Agri_Process,Conjunction,Agricultural,O
    Soil pollution,Agri_Pollution,Caused_By,horticultural practices,Agri_Process
    Contaminating farmlands,O,Caused_By,horticulture,Agri_Process
    Soil contamination,Agri_Pollution,Caused_By,horticultural activities,Agri_Process
    Polluting the earth,O,Caused_By,gardening,O
    Soil degradation,Agri_Pollution,Caused_By,horticultural methods,Agri_Process
    contaminating the soils,Agri_Pollution,Caused_By,horticultural,Agri_Process
    soil erosion,O,Caused_By,horticultural practices,Agri_Process
    deposits,O,Caused_By,horticultural activities,Agri_Process
    silt,O,Caused_By,horticultural impact,Agri_Process
    particle sediments,O,Caused_By,horticultural methods,Agri_Process
    sediments,O,Caused_By,horticultural,Agri_Process
    Polluting the land,O,Caused_By,Agricultural,O
    Soil contamination,Agri_Pollution,Caused_By,Agricultural,O
    Contaminating the farmland,O,Caused_By,Agricultural,O
    Land pollution,O,Caused_By,Agricultural,O
    Soil degradation,Agri_Pollution,Caused_By,Agricultural,O
    contaminating the soils,Agri_Pollution,Caused_By,Agricultural,O
    sedimentation,O,Caused_By,Farming practices,O
    deposits,O,Caused_By,Agricultural activities,O
    silt,O,Caused_By,Agriculture,Agri_Process
    alluvium,Crop,Caused_By,Farming,O
    settled particles,O,Caused_By,Agricultural practices,O
    sediments,O,Caused_By,Agricultural,O
    Agri_Pollution,Agri_Process,Includes,natural and synthetic fertilizers,O
    Agricultural pollutants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollution,Agri_Process,Includes,natural and chemical fertilizers,O
    Agricultural contaminants,Agri_Pollution,Includes,natural and chemical fertilizers,O
    Agri_Pollution,Agri_Process,Includes,organic and chemical fertilizers,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollutants,Crop,Includes,insecticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,pest control substances,Agri_Pollution
    Agricultural pollution,O,Includes,herbicides,Chemical
    Agri_Pollutants,Crop,Includes,chemical pesticides,Chemical
    Farming contaminants,Agri_Pollution,Includes,pest management products,O
    Agricultural contaminants,Agri_Pollution,Includes,pesticides,Chemical
    Agri_Pollution,Agri_Process,Includes,weed killers,O
    Agricultural pollutants,Agri_Pollution,Includes,herbicides,Chemical
    Farm pollution,O,Includes,weed control chemicals,Chemical
    Agri_Contaminants,Chemical,Includes,herbicides,Chemical
    Agricultural pollution,O,Includes,weed management agents,Person
    Agricultural contaminants,Agri_Pollution,Includes,herbicides,Chemical
    Agri_Pollution,Agri_Process,Includes,pesticides,Chemical
    Farm pollutants,Agri_Pollution,Includes,insect repellents,Chemical
    Agri_Pollution,Agri_Process,Includes,bug killers,O
    Agricultural contaminants,Agri_Pollution,Includes,pest management chemicals,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,insecticides,Chemical
    farm residue,O,Includes,animal wastes,Agri_Waste
    plant debris,O,Includes,animal wastes,Agri_Waste
    waste material,O,Includes,animal wastes,Agri_Waste
    agricultural waste,O,Includes,animal wastes,Agri_Waste
    biodegradable waste,Agri_Waste,Includes,animal wastes,Agri_Waste
    organic matter,Natural_Resource,Includes,animal wastes,Agri_Waste
    Agri_Residue,Crop,Includes,decomposing botanical substances,O
    Organic debris,Agri_Waste,Includes,rotting vegetation,O
    Agri_Waste,Agri_Process,Includes,deteriorating flora matter,O
    Organic material,Natural_Resource,Includes,decayed plant elements,O
    Agricultural waste,O,Includes,decomposing plant material,O
    organic matter,Natural_Resource,Includes,decaying plant material,O
    irrigation residues,Agri_Waste,Includes,salts,O
    irrigation waste,O,Includes,trace elements,O
    irrigation leftovers,O,Includes,trace metals,O
    farming residues,O,Includes,trace elements,O
    agricultural waste,O,Includes,trace metals,O
    irrigation byproducts,Agri_Waste,Includes,trace elements,O
    irrigation residues,Agri_Waste,Includes,trace metals,O
    irrigation residues,Agri_Waste,Includes,microorganisms,Organism
    Phosphorus,Nutrient,Synonym_Of,P,O
    Nutrient,Nutrient,Synonym_Of,Phosphorus,Nutrient
    Essential element,O,Synonym_Of,Phosphorus,Nutrient
    Mineral,Natural_Resource,Synonym_Of,Phosphorus,Nutrient
    Plant nutrient,Nutrient,Synonym_Of,Phosphorus,Nutrient
    P,O,Synonym_Of,Phosphate,Nutrient
    P,O,Synonym_Of,Phosphorus,Nutrient
    Nitrogenous,O,Helps_In,crop production,O
    Nitrogen element,Nutrient,Helps_In,yield of crops,O
    Nitrogen-rich,Nutrient,Helps_In,agricultural output,O
    Nitrate,Nutrient,Helps_In,crop yield,O
    Nitrogen fertilizer,Nutrient,Helps_In,harvest productivity,O
    nitrogen,Nutrient,Helps_In,crop yield,O
    P,O,Helps_In,crop production,O
    phosphorus,Natural_Resource,Synonym_Of,P,O
    P,O,Synonym_Of,phosphorus,Natural_Resource
    Polluting the soil,O,Caused_By,horticulture,Agri_Process
    Soil contamination,Agri_Pollution,Caused_By,horticultural practices,Agri_Process
    Soil pollution,Agri_Pollution,Caused_By,horticulture,Agri_Process
    Contaminating the land,O,Caused_By,horticultural activities,Agri_Process
    Soil adulteration,Agri_Pollution,Caused_By,horticulture,Agri_Process
    contaminating the soils,Agri_Pollution,Caused_By,horticultural,Agri_Process
    sedimentary deposits,O,Caused_By,horticultural activities,Agri_Process
    sediment,O,Caused_By,horticultural practices,Agri_Process
    sedimentation,O,Caused_By,horticultural impacts,Agri_Process
    sediments,O,Caused_By,horticultural,Agri_Process
    Polluting the farmland,O,Caused_By,industrial activities,O
    Soil contamination,Agri_Pollution,Caused_By,industrial activities,O
    Tainting the soil,O,Caused_By,industrial activities,O
    Contaminating the cropland,O,Caused_By,industrial activities,O
    Pollution of agricultural soils,O,Caused_By,industrial activities,O
    contaminating the soils,Agri_Pollution,Caused_By,industrial activities,O
    Agri_Pollution,Agri_Process,Caused_By,manufacturing operations,O
    Sedimentary deposits,O,Caused_By,industrial processes,O
    Agri_Pollutant,Crop,Caused_By,industrial activities,O
    Sediment,O,Caused_By,industrial operations,O
    Agri_Pollution,Agri_Process,Caused_By,industrial practices,O
    sediments,O,Caused_By,industrial activities,O
    Agri_Pollution,Agri_Process,Includes,natural and synthetic fertilizers,O
    Agricultural pollutants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollutants,Crop,Includes,natural and synthetic fertilizers,O
    Farm pollutants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agricultural pollution,O,Includes,natural and synthetic fertilizers,O
    Agricultural contaminants,Agri_Pollution,Includes,organic and inorganic fertilizers,Chemical
    Agri_Pollution,Agri_Process,Includes,insecticides,Chemical
    Agricultural pollutants,Agri_Pollution,Includes,pesticides,Chemical
    Farming pollution,O,Includes,pest control substances,Agri_Pollution
    Agri_Contaminants,Chemical,Includes,chemical pesticides,Chemical
    Agricultural pollution,O,Includes,herbicides,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,pesticides,Chemical
    Agri_Pollutants,Crop,Includes,weed killers,O
    Farming pollutants,Agri_Pollution,Includes,herbicides,Chemical
    Agricultural pollution,O,Includes,herbicides,Chemical
    Agri_Contaminants,Chemical,Includes,weed killers,O
    Farm pollutants,Agri_Pollution,Includes,herbicides,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,herbicides,Chemical
    Agricultural contaminants,Agri_Pollution,Includes,insecticides,Chemical
    Agri_Pollution,Agri_Process,Includes,organic material,Natural_Resource
    Agricultural pollutants,Agri_Pollution,Includes,organic matter,Natural_Resource
    Agri_Pollution,Agri_Process,Includes,organic matter,Natural_Resource
    Agricultural pollution,O,Includes,organic material,Natural_Resource
    Agricultural contaminants,Agri_Pollution,Includes,organic matter,Natural_Resource
    Agri_Pollutants,Crop,Includes,irrigation remains,O
    Agricultural pollutants,Agri_Pollution,Includes,irrigation remnants,O
    Agri_Pollutants,Crop,Includes,irrigation leftovers,O
    Agricultural contaminants,Agri_Pollution,Includes,irrigation residues,Agri_Waste
    Agri_Pollution,Agri_Process,Includes,irrigation waste,O
    Agricultural contaminants,Agri_Pollution,Includes,irrigation residues,Agri_Waste
    Agri_Waste,Agri_Process,Includes,animal byproducts,Agri_Waste
    Organic materials,Natural_Resource,Includes,animal wastes,Agri_Waste
    Agricultural waste,O,Includes,animal excreta,O
    Farm waste,O,Includes,animal residues,Agri_Waste
    Bio waste,O,Includes,animal byproducts,Agri_Waste
    organic matter,Natural_Resource,Includes,animal wastes,Agri_Waste
    Agri_Residue,Crop,Includes,decaying vegetation,O
    Organic waste,Natural_Resource,Includes,deteriorating plant material,O
    Biodegradable waste,Chemical,Includes,rotting botanical matter,O
    Agri_Waste,Agri_Process,Includes,decomposing plant material,O
    Organic material,Natural_Resource,Includes,decayed plant substance,Chemical
    organic matter,Natural_Resource,Includes,decaying plant material,O
    irrigation residues,Agri_Waste,Includes,salts,O
    Agri_Waste,Agri_Process,Includes,trace elements,O
    Irrigation waste,Agri_Process,Includes,trace metals,O
    Agricultural residues,Agri_Waste,Includes,trace metals,O
    Farm waste,O,Includes,trace elements,O
    Irrigation remnants,Agri_Process,Includes,trace metals,O
    irrigation residues,Agri_Waste,Includes,trace metals,O
    """

# Create the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,  # Set a reasonable limit for output tokens
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
    safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    },
    system_instruction=f"""{data}
         
    Please help me answer questions related to this dataset.Ignore the 
    tail label for those which have O. return with all the possible answers. 
    i will give some questions based on which extract the keywords from the question, 
    and fetch the relation along with the answer based on the keyword.
    make sure the answer is direct and exactly all answers should be given NOTHING SHOULD BE LEFT, take some time but scan the csv perfectly.
    i dont need to see relations of the entities.
    make sure the answers are only from dataset given and if multiple matches give all.
    Give the answer in a paragraph manner!
    Make sure the answers are unique.
    Also make a cypher query for knowledge. lets say you get the entity as the keyword FROM THE QUESTION and you identify the relation, the code would be:
    MATCH (e {{name: entity}})-[r:relation]->(related)
    RETURN e,r,related
    make sure the arrow: -> is correctly pointed everytime it may not be towards related sometime it can be away too
    
    just give output for cypher as:
    Cypher code is:
    (the code)
    and dont include ``` this is in the answer
    If there is no match, do not generate cypher code, just print No cypher code for the above answer.
    Now, if i ask the question is whatever language, 
    give the answer in that language too
    only for cypher: entity name will be translated to 
    english only
    """
    
)


# Initialize conversation history
history = []

def answer_question(question):
    question = question.lower()
    
    entity1 = None
    entity2 = None

    if "relation" in question:
        parts = question.split("between")
        if len(parts) > 1:
            entities = parts[1].strip().split("and")
            if len(entities) == 2:
                entity1 = entities[0].strip()
                entity2 = entities[1].strip()
    
    if entity1 and entity2:
        result = data[(data['entity1'].str.lower() == entity1) & (data['entity2'].str.lower() == entity2)]
        
        if not result.empty:
            return result['relation'].tolist()
        else:
            return f"No relation found between {entity1} and {entity2}."
    else:
        return "Please provide a valid question format."

print("Bot: Hello!!!\n")
while True:
    user_input = input("You: ")
    
    # Check for exit condition
    if user_input.lower() in ['exit', 'quit']:
        print("Bot: Goodbye!")
        break

    # Answer the question using the custom function
    model_response = answer_question(user_input)
    print(f'Question: {user_input}')
    # If no response, use the model to generate a response
    if isinstance(model_response, str):
        chat_session = model.start_chat(history=history)
        try:
            response = chat_session.send_message(user_input)
            model_response = response.text
        except generation_types.StopCandidateException as e:
            model_response = f"Error: {e}"


        print(f'Bot: {model_response}')
        print()
    
# history.append({"user_input": user_input, "model_response": model_response})
# print(history)
    
    

