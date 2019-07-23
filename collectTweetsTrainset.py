import twint
import sqlite3

def runTwint(username):
    try:
        print(username)
        c = twint.Config()
        # c.Store_csv = True
        c.Limit = 200
        # c.Count = True
        # c.Output = "20190120_"+username
        c.Database = "trainingSet_79.db"
        # c.Username = username
        c.User_id = username
        # c.Search = username
        # c.Format = "Tweet id: {id} | Date: {date}| Time: {time}| Username: {username} | hashtag: {hashtags}"
        # c.User_full = True
        # c.Profile_full = True
        # 1
        # twint.run.Lookup(c)
        twint.run.Profile(c)
    except:
        return False
    return True


users = ["kartikums","luthfiayuri","Febrianadiah","cahniode","noviarizkym","Aidhaasri","fazaharahap","CKY8","Rule_ryuken","Alfi_trb70","farras_cp","Rullesmana","Bononmana","dwi_arsyandi","Thirzaria","harnendio","rifqiat123","fruzciante","zarkasino","zakasandra","erwnprstyo","ndahendach","ralishanabila","ilanaina","graciously_","karinrahman","girnadianis","J_wonwo0","Jorhutagalung","dindasenja","andinrahmania","anjave_","zakyfarism","Fardhian","Anggilani","Isyawads","rojulpj","tatitya","alyaizaaz","ziahanum","Yesftrrr","Raraso","amaulana","jengasri","marshawindira","maynifique","dewisendang","cynt2","zam_arz","Edwinakustiawan","rarapristine","aninditastywt","kendydanangp"]
# users2 = ["827anh", "acul_SR", "adimpil", "akirafujimari", "akoor", "Alfi_trb70", "almo_kemal", "alyaizaaz", "amaulana", "aMrazing", "anawid_", "andrysatrio", "aninditastywt", "anjave_", "anrisapriyasta", "ardiantoCahyadi", "ariefrian_", "arioadimas", "AstariNPianist", "Atha_Hey", "bagus_kr", "bayu_joo", "bononmana", "BrianTiZi", "cahniode", "chiiky_meong2", "CKY8", "dewisendang", "dillandud", "dimazdmagan", "djaycoholyc", "dyahsrahayu", "dylla_Eze", "Edwinakustiawan", "ekkyfsn", "erwnprstyo", "faradinayu", "fardhian", "farrasCP", "fazaharahap", "fruzciante", "gilkoonk", "girnadianis", "gustiraganata", "harnendio", "Heskiey", "Hujandisenja", "Iffatsatrya", "ilanaina", "ItafianaLina", "ivanlanin", "izzxtid"]
users3 = ["cahniode","ZakaSandra","jefrinichol","bayu_joo","farrasCP","karinrahman","dimazdmagan","nathaliamwt","Hujandisenja","sarahsiregar","aMrazing","arioadimas","seterahdeh","aninditastywt","risvim","adimpil","anawid_","ivanlanin","zaharanissa","pinotski","maynifique","girnadianis","sherlykarakawa","djaycoholyc","bononmana","fardhian","ntans","Atha_Hey","andrysatrio","ochaganneosha","kendydanangp","PancaNM","Raraso","sittakarina","jokoanwar","izzxtid","NadhiraKamilah","nblfauzia","gustiraganata","denty_kusuma","nuzuliramadhina","nilamtriarmy","RFN19","ItafianaLina","NurizalAR","acul_SR","kalvid","zulfikarrifaldy","yayattttt","sienput","ardiantoCahyadi","YusrilArief","akoor","Melkikun","jokidz90","mtaufiqasmara","ekkyfsn","deprayoga","827anh","ariefrian_","pujianto_7","FanulDoang","rizkiriko","Heskiey","chiiky_meong2","dylla_Eze","YonasLiem","hapsorohap","dessyarii","almo_kemal","WaOnEmperoR","gerrymaulana182","ninis89","akirafujimaru","bagus_kr","AstariNPianist","PambudiSurya","rajib_saputra","anrisapriyasta","sandi_rizky_k","Marsya1989","faradinayu","ViscoRedz","gilkoonk","dyahsrahayu","alyaizaaz","AlkindiQnd","munjiahnur","BrianTiZi","putriks","FannyAOctaviana","Iffatsatrya","IsyawaDS","fatimy","Rule_ryuken","KupRilL","dillandud","lidyaocta"]

#update 21/03/2019 from https://www.sociabuzz.com/client/influencer/page/5/?media=twitter&agemin=30
users4 = ["tiandarinie","april_hamsa","tuty_utut","nunainun","rizkyardinsyah","akunovitania","anis_sa_ae","Justephanie","HerdianaHS","ItsEizy","triwidyatmoko","cintaramlanunes","beautyasti1","Mugniar","AncaSyah","amranhamdani","HenySurockboyo","novanovili","LizaFathia","sejuta_cinta","suzannita","vik_ww","DuniaBiza","nenglisojung","yunitatriyana","klikarbain","endikkoeswoyo","ShemMD","PrincessBrigita","disyadimi","nenghujan","DennyDenox","hermavvan","zataligouw","astridhere","dodon_jerry","catperku","ariemohr","virustraveling","ummihasfa","KorneliaLuciana","felyina","MuntilanKu","sloppypoppy","cputriarty","sophie_tobelly","aimanricky","Andika_dTT","irsyachendikia","ridwanfanwar","aqbastian","addiems","vitamasli","DESSYilsanty","Dzawinur"]

users5 = ["benakribo","AdheTora","andreOPA","PrincessBrigita","iamsolisoul","arioadimas","veanarvian","febrikholid","fendychow","omdirects_","sittakarina","denniserfindo","dimazdmagan","iamMariza","tianlustiana","BrianSO7","irfanaon","LizaFathia","bayu_joo","endikkoeswoyo","arievrahman","wendriiy","chachathaib","falla_adinda","catperku","AryMozta","ficofachriza_","rasarab","katadochi","zarryhendrik","astarianadya","chikarein","Naufalfitraa","miss7sins","cputriarty","Aziz_Ngok","Raffinagita1717","grace_melia","SekseehOnline","idhoidhoid","lianny","NovaWijaya94","vitamasli","nenkmanda","HerdianaHS","dinidinda","Mugniar","newendi","dewyang_","DIrtyHarryHNc","sarahazkaa","HEYDEERAHMA","NasyaMarcella","patrishiela","bangkitbangkiit","pupututami","aimanricky","inarahsyarafina","ifyalyssa","rezaphlv","nabilabull","dewirieka","novanovili","nenghujan","Princess_Rhie","tuty_utut","leonagustine","Nona_HitamPahit","tettytanoyo","ShemMD","Praz_Teguh","Dzawinur","sophie_tobelly","dhanypramata","suciutami","haykalkamil","KennyAuztin","dayatpiliang","vik_ww","dodon_jerry","noormafmz","ummihasfa","mutmuthea","april_hamsa","arfisulthani","khairinadiar","_febrian","yudhakhel","nitasofiani","difasabila","DuniaBiza","anis_sa_ae","yukianggia","cindykarmoko","sikonyols","virustraveling","dndea","HaZetZet","sishabiq","Frimawan","astrityas_","sloppypoppy","umimita","KoperTraveler","shitlicious","ibnu_nugraha_","alexxrex","stellalee92","rizal_monzterz","labollatorium","ariemohr","Noviyanashiali","AdityaSani","nadiasoekarno","hendralm","NoniZara","chawrelia","aMrazing","yurimuthmainnah","suzannita","xxRynisme","FellyYoung21","tentangrinda","dindayularasati","rrmahayukd","felyina","MuntilanKu","mrizagg","ladyfaustine","vannysariz","harivalzayuka","robbypurba","feliciangelica","foodyfloody","chintyatengens","viratandia","RafelAdrian","daintykr","mariokacang","cyndaadissa","ayunda_leni95","official_evra","GeraNgidolKak","neyrhiza","yoraanastasha","zarlinda","chrismathbe","amandatydes","rizalmuk","BayuRistian","rahneputri","ririekayan","zataligouw","cicidesri","willchuun","Andika_dTT","Ayuditha18","GilangDirga","ilayahya_","Debbydabby","zyzyladybluee","Mistfast","arif_alfiansyah","atikabidin","nenengggND","yudidam","Sourcherie","ikram_AFRO","ItsEizy","APITOOO","AncaSyah","BillySuryaD","Thania_Grey","niputuchandra","ridwanfanwar","akunovitania","titah_player05","GandhiFernando","alexandraelma","mewpawmily","Grestyan","renitarifdah","AyoJalan2","YolandaVebyola","akwibowo","SalwaFila","HdAdjie","aryadega","GurunMzi3","faizsadad_","artdiles","Naufalnurhilmi3","meisitawulandar","LALAkarmela","jayakabajay","DennyDenox","edhozell","Hujandisenja","widyatarmizi","carimichan","3TataChiBi","TheSilver_Sword","BobbyTakara","gheasafferina","ihsantarore","theRosb","keirashabira","risafitriam","baimrahmat_","shesNearandFar","SafaniaRomas","laiqulfakhri","alvansepty","bhimawibawa","kepinhelmy","dhonypermana","MaXnuum","klikarbain","gerry_giovanny","DESSYilsanty","Celaoo","dyanputri","MrWahyuPrasetya","MerryRiana","iyamrenzia","triwidyatmoko","leejdy","disyadimi","medyrenaldy_","angelinyangg","irsyachendikia","JenniferOdelia","nvtamlia","arnoldyesontha","aqbastian","abetedefghijklm","enoomomsen","astridhere","annisavigne","Ana__Livian","reynaldirifaldo","adilafitri","YulioChandra","chimchim_busan","eldwen_hjs","vabyramauriz","andileolim95","elma_amellia","yunitatriyana","HenySurockboyo","beautyasti1","NurDalilahPutr","ijoyhatta","BertiRyan","remaulian","HannaSutiono_","cintaramlanunes","MIRATIPRIMASARI","Desty007"]
users6 = ['AriaGagan','Nunu rohmanu','Nandayanishanti','tiwitico','jeihanm','1u__16','sapiEmoh','indramaars','unkosmyk','Dini miati','Mayaw','Harpa Ranusono','roro soekamto','Andika31','Helia yuliartin','Majachandra','rifkychaerul','anjarseno','inov Nalle','Tigor73742372','arisman_98220','Qonna_30','chipachiko','Yasin','ibnu ','kodarsyah','Sigitmuantaf','g4givangkara','daniel_rakian','lilisazami','Rifqiat123','Lutfi','OchengSuhendraP','nanda_mamen','andhikavikram','Bakrie','andi_ady','Cesaria','ARTHUR ADYTYA PRATAMA','aniez_marana','julizar','Elmarozi','Abbhakbar','Muliati','Agatharatu01','irniaty','MustariFauzi','Erwin','Rivai.r','Zulfiqr02826085','Rasul 08','Alamsyah75851220','Zulfiqr02826085','Mardiana','Ridwan.k','Rismala','@ewinvespa','Ridwani','Risnawaty','Saharuddin','Akbar.hasan','Rahmat.saeni','Marlina','Marlina','slamet','eross_ion','Thamrin usman','Thamrin usman','Janwar','Poppy','Sriyanti Busso','Kiki Riski','ATNKCR','@airah262014','MustariFauzi','ThamrinUsman2','Farah','abrar','Husnialiah husnialiah']

# for user in users6:
#     runTwint(user)


con3 = sqlite3.connect("trainingSet_79 - Copy.db")
cur = con3.execute("SELECT id, id_str, name, username FROM users ")
for row in cur:
    print("USERNAME: "+row[3])
    runTwint(row[0])
# runTwint('AriaGagan')