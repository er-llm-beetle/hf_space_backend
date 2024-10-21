# from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

import re, os
import pandas as pd
from openai import OpenAI

from dotenv import load_dotenv




def get_answer_multiple_choice_w_dstype(question, options, model, num_fewshot, dstype, context = None, tokenizer=None, api=False, gguf=False, repo_id=None):

    # generation_config = model.generation_config
    # generation_config.max_length = 1000
    # generation_config.pad_token_id = tokenizer.pad_token_id


    if dstype == 'tc':
        try:
            formatted_options = options.replace(", ", ",\n")
            print("OPTIONS:\n", formatted_options)
            
            options = formatted_options
        except Exception as e:
            print(f"No options - Error: {e}")
            options = 'No options - Error'

        messages = [
            {
                "role": "system",
                "content": "You are an AI designed to answer questions in Azerbaijani. Your task is to select the correct option from the given question and answer choices. You are given a statement along with multiple options that represent different topics. Choose the option that best categorizes the statement based on its topic. Choose the single letter (A, B, C, D, E, F, G, H, I, J) that best answers the question. Respond with only the letter of the chosen answer, without any additional text."
            },
            {
                "role": "user",
                # "content": "Question:\nMənə yaxın filial tapmağa kömək edin.\nOptions:\nA) ATM,\nB) FEES,\nC) OTHER,\nD) CARD,\nE) ACCOUNT,\nF) TRANSFER,\nG) PASSWORD,\nH) LOAN,\nI) CONTACT,\nJ) FIND\n\nAnswer:"
                "content": "Question:\nMənə yaxın filial tapmağa kömək edin.\nOptions:\nA) BANKOMAT,\nB) KOMİSSİYA,\nC) DİGƏR,\nD) KART,\nE) HESAB,\nF) KÖÇÜRMƏ,\nG) ŞİFRƏ,\nH) KREDİT,\nI) ƏLAQƏ,\nJ) TAP\n\nAnswer:" # Reshad's updated data

            },
            {
                "role": "assistant",
                "content": "J"
            },
            {
                "role": "user",
                # "content": "Question:\nŞifrə təyin etməliyəm, bunu necə edə bilərəm?\nOptions:\nA) ATM,\nB) FEES,\nC) OTHER,\nD) CARD,\nE) ACCOUNT,\nF) TRANSFER,\nG) PASSWORD,\nH) LOAN,\nI) CONTACT,\nJ) FIND\n\nAnswer:"
                "content": "Question:\nŞifrə təyin etməliyəm, bunu necə edə bilərəm?\nOptions:\nA) BANKOMAT,\nB) KOMİSSİYA,\nC) DİGƏR,\nD) KART,\nE) HESAB,\nF) KÖÇÜRMƏ,\nG) ŞİFRƏ,\nH) KREDİT,\nI) ƏLAQƏ,\nJ) TAP\n\nAnswer:" # Reshad's updated data
            },
            {
                "role": "assistant",
                "content": "G"
            },
            {
                "role": "user",
                # "content": "Question:\nMən virtual kart almaq istəyirəm. Necə əldə edə bilərəm?\nOptions:\nA) ATM,\nB) FEES,\nC) OTHER,\nD) CARD,\nE) ACCOUNT,\nF) TRANSFER,\nG) PASSWORD,\nH) LOAN,\nI) CONTACT,\nJ) FIND\n\nAnswer:"
                "content": "Question:\nMən virtual kart almaq istəyirəm. Necə əldə edə bilərəm?\nOptions:\nA) BANKOMAT,\nB) KOMİSSİYA,\nC) DİGƏR,\nD) KART,\nE) HESAB,\nF) KÖÇÜRMƏ,\nG) ŞİFRƏ,\nH) KREDİT,\nI) ƏLAQƏ,\nJ) TAP\n\nAnswer:" # Reshad's updated version
            },
            {
                "role": "assistant",
                "content": "D"
            },
            {
                "role": "user",
                # "content": "Question:\nNə qədər vaxt tələb olunur ki, nağd pul çıxarışı görünsün?\nOptions:\nA) ATM,\nB) FEES,\nC) OTHER,\nD) CARD,\nE) ACCOUNT,\nF) TRANSFER,\nG) PASSWORD,\nH) LOAN,\nI) CONTACT,\nJ) FIND\n\nAnswer:"
                "content": "Question:\nNə qədər vaxt tələb olunur ki, nağd pul çıxarışı görünsün?\nOptions:\nA) BANKOMAT,\nB) KOMİSSİYA,\nC) DİGƏR,\nD) KART,\nE) HESAB,\nF) KÖÇÜRMƏ,\nG) ŞİFRƏ,\nH) KREDİT,\nI) ƏLAQƏ,\nJ) TAP\n\nAnswer:" # Reshad's updated version
            },
            {
                "role": "assistant",
                "content": "A"
            },
            {
                "role": "user",
                # "content": "Question:\nSalam, bu gün kripto ilə balans artırmağa çalışdım, amma alınmadı. Ola bilərmi ki, istifadə etdiyim bankın növü və ya mənim olduğum ölkə ilə əlaqədardır? Hesabımdan da pul çıxıb.\nOptions:\nA) ATM,\nB) FEES,\nC) OTHER,\nD) CARD,\nE) ACCOUNT,\nF) TRANSFER,\nG) PASSWORD,\nH) LOAN,\nI) CONTACT,\nJ) FIND\n\nAnswer:"
                "content": "Question:\nSalam, bu gün kripto ilə balans artırmağa çalışdım, amma alınmadı. Ola bilərmi ki, istifadə etdiyim bankın növü və ya mənim olduğum ölkə ilə əlaqədardır? Hesabımdan da pul çıxıb.\nOptions:\nA) BANKOMAT,\nB) KOMİSSİYA,\nC) DİGƏR,\nD) KART,\nE) HESAB,\nF) KÖÇÜRMƏ,\nG) ŞİFRƏ,\nH) KREDİT,\nI) ƏLAQƏ,\nJ) TAP\n\nAnswer:"    # Reshad's updated version
            },
            {
                "role": "assistant",
                "content": "E"
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\nOptions:\n{options}\n"
            }
        ]



    elif dstype == 'mmc':
        try:
            formatted_options = options.replace(", ", ",\n")
            print("OPTIONS:\n", formatted_options)
            
            options = formatted_options
        except Exception as e:
            print(f"No options - Error: {e}")
            options = 'No options - Error'

        messages = [
            {
                "role": "system",
                "content": "You are an AI designed to solve mathematical word problems in Azerbaijani. Your task is to analyze the given question and select the correct option from the provided choices. Choose the single letter (A, B, C, D, E) that best answers the question. Respond with only the letter of the chosen answer, without any additional text."
            },
            {
                "role": "user",
                "content": "Question:\nMeyvə-tərəvəz satıcısı 15 kq yerkökü, 13 kq badımcan və 8 kq kartof satışa qoydu. Onlardan yalnız yarısını satdı. O, nə qədər tərəvəz satdı?\nOptions:\nA) 18,\nB) 20,\nC) 23,\nD) 16,\nE) 24\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "A"
            },
            {
                "role": "user",
                "content": "Question:\nXəzər ilk saatda 2 kilometr yol getdi. Əgər o, ikinci saatda iki dəfə daha uzağa getsə, iki saat ərzində nə qədər yol qət edərdi?\nOptions:\nA) 24,\nB) 36,\nC) 48,\nD) 12,\nE) 6\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "E"
            },
            {
                "role": "user",
                "content": "Question:\nBir riyaziyyat testində Mark ən az balın iki qatını topladı. Ən yüksək bal 98, balların fərqi isə 75 olarsa, Markın balı nə qədər idi?\nOptions:\nA) 23,\nB) 46,\nC) 49,\nD) 98,\nE) 36\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "B"
            },
            {
                "role": "user",
                "content": "Question:\nBu gün məktəbdə keçirilən elm dərsində Cek öyrəndi ki, qalaktikada hər bir planet üçün 8 əlavə günəş sistemi mövcuddur. Əgər qalaktikada 20 planet varsa, ümumilikdə neçə günəş sistemi və planet vardır?\nOptions:\nA) 160,\nB) 180,\nC) 200,\nD) 220,\nE) 240\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "C"
            },
            {
                "role": "user",
                "content": "Question:\nBir stəkan 250 ml-ə bərabərdir. Brian özünə, həyat yoldaşına, iki uşağına, valideynlərinə və həyat yoldaşının valideynlərinə lazanya hazırlayır. Reseptə görə hər porsiya üçün 1/2 stəkan süd tələb olunur. Hər bir insanın 2 porsiya yeməsi gözlənilirsə, Brian neçə 1 litrlik süd qabı almalıdır?\nOptions:\nA) 1,\nB) 2,\nC) 3,\nD) 4,\nE) 5\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "B"
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\nOptions:\n{options}\n"
            }
        ]



    elif dstype == 'mc': 

        try:
            split_options = re.split(r'(?<=,)\s*(?=[A-D]\))', options)

            # Format each option
            formatted_options = []
            for opt in split_options:
                # Remove leading/trailing whitespace
                opt = opt.strip()
                # Ensure there's a comma at the end if not already present
                if not opt.endswith(','):
                    opt += ','
                formatted_options.append(opt)

            # Join with newlines
            result = '\n'.join(formatted_options)
            print("OPTIONS:\n", result)
            
            # formatted_options = options.replace(", ", "\n") # v1.2
            # print("OPTIONS: ", formatted_options)
            
            options = result
        except Exception as e:
            print(f"No options - Error: {e}")
            options = 'No options - Error'

        messages = [
            {
                "role": "system",
                "content": "You are an AI that selects the most accurate answer in Azerbaijani based on a given question. You will be provided with a question in Azerbaijani and multiple options in Azerbaijani. Choose the single letter (A, B, C, D) that best answers the question. Respond with only the letter of the chosen answer, without any additional text."
            },
            {
                "role": "user",
                "content": "Question:\nBank krediti haqqında deyilənlərdən hansı yalnışdır?\nOptions:\nA) Bağlanmış müqaviləyə uyğun olaraq qaytarılmalıdır\nB) Müddətin uzaldılması şərtinin müqavilədə olması zəruridir\nC) Təminatsız verilə bilər\nD) Pul vəsaitinin verilməsinə dair qarantiya və zəmanət öhdəlikləri kredit anlayışına aid deyil.\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "D"
            },
            {
                "role": "user",
                "content": "Question:\nDepozit haqqında deyilənlərdən hansı yalnışdır?\nOptions:\nA) Depozit müqaviləsi 0 (sıfır) faiz dərəcəsi ilə bağlana bilməz\nB) Tələbli və ya müddətli ola bilər\nC) Cari hesabda saxlanılan vəsait də depozit sayılır\nD) Müştərinin tələbi ilə qaytarılmalıdır.\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "A"
            },
            {
                "role": "user",
                "content": "Question:\nİpoteka krediti ilə bağlı hansı ifadə doğrudur?\nOptions:\nA) İpoteka krediti mütləq əmlak zəmanəti ilə verilir\nB) İpoteka krediti yalnız maşın almaq üçün istifadə edilə bilər\nC) İpoteka krediti heç bir faiz ödənilmədən verilə bilər\nD) İpoteka krediti yalnız banklar tərəfindən təqdim olunur.\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "A"
            },
            {
                "role": "user",
                "content": "Question:\nAşağıdakı müddəalardan hansı yalnışdır\nOptions:\nA) tələbələrə kredit verilə bilər, B) borcalan (gəliri muzdlu işlə əlaqədar olan) ən azı son 6 (altı) ay üzrə aylıq gəlirlərini təsdiq edən sənəd təqdim etməlidir, C) təminatlı aktiv - təminatı məqbul təminat olduqda onun bazar dəyəri təmin etdiyi öhdəlik məbləğinin 100 (yüz) faizindən, təminat məqbul təminat olmadıqda isə 150 (yüz əlli) faizindən az olmayan aktiv, D) Məqbul təminat bankın daxili qaydalarına əsasən müəyyən edilir\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "D"
            },            
            {
                "role": "user",
                "content": "Question:\nVergi orqani ilə aparılan əməliyyatlarda bankların məsuliyyətinə dair qeyd edilənlərdən biri doğru deyil\nOptions:\nA) Banklar hüquqi şəxslərə, onların filial və nümayəndəliklərinə, fərdi sahibkarlara yalnız onlar vergi orqanı tərəfindən verilən şəhadətnamə-dublikatını təqdim etdikdən sonra bank hesabları açırlar. Şəhadətnamə-dublikat alındığı tarixdən 10 gün müddətində hesabın açılması üçün istifadə olunmadıqda qüvvədən düşmüş hesab edilir., B) Vergi borcunun bank hesabinda dondurulması tələbi gəldikdə bank müştərinin milli və ya xarici valyutada cari və ya digər hesablarında borc məbləğinin 105 faizi həcmində pul vəsaiti üzrə məxaric əməliyyatları məhdudlaşdırır, C) Vergi ödəyicisinin milli və ya xarici valyutada cari və ya digər hesablarında vəsait olduqda vergi orqanının alınmalı sərəncamı bank və ya digər kredit təşkilatı tərəfindən 3(üç) gün ərzində icra edilir., D) Sərəncam gəldiyi zaman müştərinin hesabinda vəsait olmadığı və qismən kifayət etdiyi halda sərəncam bankda saxlanılır, vəsait daxil olduqca icra edilir\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "C"
            },            
            {
                "role": "user",
                "content": f"Question:\n{question}\nOptions:\n{options}\n"
            }
        ]

        # aze
        # messages = [
        #     {
        #         "role": "system",
        #         "content": "Sən Azərbaycan dilində verilmiş suallara əsaslanaraq ən dəqiq cavabı seçən süni intellektsən. Sənə verilmiş sualdan və seçimlərdən yola çıxaraq, ən uyğun olan tək hərfi (A, B, C, D) seç. Yalnız seçilmiş cavabın hərfini, heç bir əlavə mətni olmadan cavab ver."
        #     },
        #     {
        #         "role": "user",
        #         "content": "Sual:\nBank krediti haqqında deyilənlərdən hansı yalnışdır?\nSeçimlər:\nA) Bağlanmış müqaviləyə uyğun olaraq qaytarılmalıdır\nB) Müddətin uzaldılması şərtinin müqavilədə olması zəruridir\nC) Təminatsız verilə bilər\nD) Pul vəsaitinin verilməsinə dair qarantiya və zəmanət öhdəlikləri kredit anlayışına aid deyil.\n\nCavab:"
        #     },
        #     {
        #         "role": "assistant",
        #         "content": "D"
        #     },
        #     {
        #         "role": "user",
        #         "content": "Sual:\nDepozit haqqında deyilənlərdən hansı yalnışdır?\nSeçimlər:\nA) Depozit müqaviləsi 0 (sıfır) faiz dərəcəsi ilə bağlana bilməz\nB) Tələbli və ya müddətli ola bilər\nC) Cari hesabda saxlanılan vəsait də depozit sayılır\nD) Müştərinin tələbi ilə qaytarılmalıdır.\n\nCavab:"
        #     },
        #     {
        #         "role": "assistant",
        #         "content": "A"
        #     },
        #     {
        #         "role": "user",
        #         "content": "Sual:\nİpoteka krediti ilə bağlı hansı ifadə doğrudur?\nSeçimlər:\nA) İpoteka krediti mütləq əmlak zəmanəti ilə verilir\nB) İpoteka krediti yalnız maşın almaq üçün istifadə edilə bilər\nC) İpoteka krediti heç bir faiz ödənilmədən verilə bilər\nD) İpoteka krediti yalnız banklar tərəfindən təqdim olunur.\n\nCavab:"
        #     },
        #     {
        #         "role": "assistant",
        #         "content": "A"
        #     },
            # {
            #     "role": "user",
            #     "content": "Sual:\nAşağıdakı müddəalardan hansı yalnışdır\nSeçimlər:\nA) tələbələrə kredit verilə bilər, B) borcalan (gəliri muzdlu işlə əlaqədar olan) ən azı son 6 (altı) ay üzrə aylıq gəlirlərini təsdiq edən sənəd təqdim etməlidir, C) təminatlı aktiv - təminatı məqbul təminat olduqda onun bazar dəyəri təmin etdiyi öhdəlik məbləğinin 100 (yüz) faizindən, təminat məqbul təminat olmadıqda isə 150 (yüz əlli) faizindən az olmayan aktiv, D) Məqbul təminat bankın daxili qaydalarına əsasən müəyyən edilir\n\nCavab:"
            # },
            # {
            #     "role": "assistant",
            #     "content": "D"
            # },            
            # {
            #     "role": "user",
            #     "content": "Sual:\nVergi orqani ilə aparılan əməliyyatlarda bankların məsuliyyətinə dair qeyd edilənlərdən biri doğru deyil\nSeçimlər:\nA) Banklar hüquqi şəxslərə, onların filial və nümayəndəliklərinə, fərdi sahibkarlara yalnız onlar vergi orqanı tərəfindən verilən şəhadətnamə-dublikatını təqdim etdikdən sonra bank hesabları açırlar. Şəhadətnamə-dublikat alındığı tarixdən 10 gün müddətində hesabın açılması üçün istifadə olunmadıqda qüvvədən düşmüş hesab edilir., B) Vergi borcunun bank hesabinda dondurulması tələbi gəldikdə bank müştərinin milli və ya xarici valyutada cari və ya digər hesablarında borc məbləğinin 105 faizi həcmində pul vəsaiti üzrə məxaric əməliyyatları məhdudlaşdırır, C) Vergi ödəyicisinin milli və ya xarici valyutada cari və ya digər hesablarında vəsait olduqda vergi orqanının alınmalı sərəncamı bank və ya digər kredit təşkilatı tərəfindən 3(üç) gün ərzində icra edilir., D) Sərəncam gəldiyi zaman müştərinin hesabinda vəsait olmadığı və qismən kifayət etdiyi halda sərəncam bankda saxlanılır, vəsait daxil olduqca icra edilir\n\nCavab:"
            # },
            # {
            #     "role": "assistant",
            #     "content": "C"
            # },            
        #     {
        #         "role": "user",
        #         "content": f"Sual:\n{question}\nSeçimlər:\n{options}\n"
        #     }
        # ]



    elif dstype == 'arc':

        try:
            # options = ['Bu, gün işığını enerjiyə çevirir.', 'Bu, hüceyrənin bütün funksiyalarını idarə edir.', 'Bu, hüceyrə boyunca suyu daşıyır.', 'Şəkəri parçalayaraq enerji buraxır.']

            options = eval(options) # from text to list
            options = ',\n'.join([f"{chr(65 + i)}) {sentence.rstrip('.')}" for i, sentence in enumerate(options)])
            print("OPTIONS:\n", options)
            
            # options = '\n'.join([f"{chr(65 + i)}) {sentence.rstrip('.')}" for i, sentence in enumerate(options)]) # v2

        except Exception as e:
            print(f"No options - Error: {e}")
            options = 'No options - Error'
    
        messages = [
            {
                "role": "system",
                # "content": "You are an AI designed to answer questions in Azerbaijani based on reasoning and knowledge. Your task is selecting the most accurate answer in Azerbaijani based on a given question. Choose the single letter (A, B, C, D) that best answers the question. Respond with only the letter of the chosen answer, without any additional text."
                "content": "You are an AI designed to answer questions in Azerbaijani based on reasoning and knowledge. Your task is to select the correct option from the given question and answer choices. You are given a question along with multiple options. Choose the correct option. Choose the single letter (A, B, C, D) that best answers the question. Respond with only the letter of the chosen answer, without any additional text."  # Reshad's update
            },
            {
                "role": "user",
                "content": "Question:\nHansı material ən yaxşı istilik izolyatorudur?\nOptions:\nA) şüşə,\nB) metal,\nC) su,\nD) ağac\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "D"
            },
            {
                "role": "user",
                # "content": "Question:\nDuzları sudan ayırmağın ən yaxşı yolu\nOptions:\nA) yağ,\nB) istilik,\nC) maqnit,\nD) əsəbləşmə alkoqolu\n\nAnswer:" # old version
                "content": "Question:\nDuzun sudan ayrılması üçün ən yaxşı yol ___ istifadəsidir\nOptions:\nA) neft,\nB) İstilik,\nC) maqnit,\nD) İsopropil spirt\n\nAnswer:" # new arc version
            },
            {
                "role": "assistant",
                "content": "B"
            },
            {
                "role": "user",
                # "content": "Question:\nTələbə təcrübəni təkrarlaya bilmək üçün hansı addımı atmalıdır?\nOptions:\nA) təcrübəni bir neçə dəfə təkrarlayın,\nB) bütün prosedurların qeydini saxlayır,\nC) Təcrübə quruluşunun bir modelini hazırlayın,\nD) Təcrübə şərtlərinin bir hissəsini dəyişdirin\n\nAnswer:" # old version
               "content": "Question:\nŞagird təcrübəni təkrarlaya bilmək üçün ilk öncə hansı addımı atmalıdır?\nOptions:\nA) Təcrübəni dəfələrlə aparmalıdır,\nB) bütün prosedurları qeydə almalıdır,\nC) Eksperimentin quruluşunun modelini hazırlamalıdır,\nD) Eksperimentin bəzi şərtlərini dəyişməlidir\n\nAnswer:" # new arc
            },
            {
                "role": "assistant",
                "content": "B"
            },
            {
                "role": "user",
                # "content": "Question:\nDDT adlı kimyəvi maddə bir vaxtlar zərərli həşəratları öldürmək üçün istifadə olunurdu. Araşdırmalar bu kimyəvi maddənin bəzi quş növləri üçün zərərli olduğunu göstərdikdə, DDT-nin istifadəsi dayandırıldı. Elmi proses DDT-nin quşlara zərər verdiyini anlamalarına necə kömək edə bildi?\nOptions:\nA) Kimyəvi maddə ilə bağlı müxtəlif proqnozlar verildi,\nB) Kimyəvi maddə ilə aparılan testlər fərqli nəticələr verdi,\nC) Kimyəvi maddə ilə təkrarlanan sınaqlar eyni nəticəyə gəldi,\nD) Kimyəvi maddənin xüsusiyyətlərini yoxlamaq üçün müxtəlif prosedurlardan istifadə edilmişdir\n\nAnswer:" # old version
                "content": "Question:\nDDT adlı bir kimyəvi maddə bir zamanlar zərərverici həşəratları öldürmək üçün istifadə edilirdi. Araşdırmalar bu kimyəvi maddənin bəzi növ quşlar üçün zərərli olduğunu göstərdiyində, DDT-nin istifadəsi dayandırıldı. Elmi proses alimlərin DDT-nin quşlara zərərli olduğunu başa düşmələrinə ən yaxşı şəkildə necə kömək edə bildi?\nOptions:\nA) Kimyəvi maddə haqqında müxtəlif proqnozlar verildi,\nB) Kimyəvi maddə üzərində aparılan testlər müxtəlif nəticələr verdi,\nC) Kimyəvi maddə üzrə təkrarlanan sınaqlar eyni nəticəni göstərdi,\nD) Kimyəvi maddənin xüsusiyyətlərini yoxlamaq üçün müxtəlif prosedurlar tətbiq edildi\n\nAnswer:" # new version
            },
            {
                "role": "assistant",
                "content": "C"
            },
            {
                "role": "user",
                # "content": "Question:\nKainatdakı bütün qalaktikalar\nOptions:\nA) çoxlu ulduzlardan ibarətdir,\nB) saatın istiqamətində dönür,\nC) Kimyəvi maddə ilə təkrarlanan sınaqlar eyni nəticəyə gəldi,\nD) eyni sayda ulduz var\n\nAnswer:"
                "content": "Question:\nSəs dalğaları və okean dalğaları hər ikisi oxşardır çünki, hər ikisi\nOptions:\nA) uzununa dalğalardır,\nB) eyni tezlikdədirlər,\nC) enerjini bir yerdən o biri yerə daşıyır,\nD) Müəyyən vasitəni bir yerdən başqa yerə aparır\n\nAnswer:" # new version
            },
            {
                "role": "assistant",
                # "content": "A"
                "content": "C"
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\nOptions:\n{options}\n"
            }
        ]

        # az
        # messages = [
        #     {
        #         "role": "system",
        #         "content": "Sən suallara Azərbaycan dilində, məntiq və bilik əsasında (based on reasoning and knowledge) cavab verən süni intellektsən. Verilmiş suala əsaslanaraq ən dəqiq cavabı seçmək sənin tapşırığındır. Suala ən yaxşı cavab verən tək hərfi (A, B, C, D) seç. Yalnız seçilmiş cavabın hərfini, heç bir əlavə mətni olmadan cavab ver."
        #     },
#             {
#                 "role": "user",
#                 "content": "Sual:\nHansı material ən yaxşı istilik izolyatorudur?\nSeçimlər:\nA) şüşə,\nB) metal,\nC) su,\nD) ağac\n\nCavab:"
#             },
#             {
#                 "role": "assistant",
#                 "content": "D"
#             },
#             {
#                 "role": "user",
#                 "content": "Sual:\nDuzları sudan ayırmağın ən yaxşı yolu\nSeçimlər:\nA) yağ,\nB) istilik,\nC) maqnit,\nD) əsəbləşmə alkoqolu\n\nCavab:" # old version
#                 # "content": "Sual:\nDuzun sudan ayrılması üçün ən yaxşı yol ___ istifadəsidir\nSeçimlər:\nA) neft,\nB) İstilik,\nC) maqnit,\nD) İsopropil spirt\n\nCavab:" # new arc version
#             },
#             {
#                 "role": "assistant",
#                 "content": "B"
#             },
#             {
#                 "role": "user",
#                 "content": "Sual:\nTələbə təcrübəni təkrarlaya bilmək üçün hansı addımı atmalıdır?\nSeçimlər:\nA) təcrübəni bir neçə dəfə təkrarlayın,\nB) bütün prosedurların qeydini saxlayır,\nC) Təcrübə quruluşunun bir modelini hazırlayın,\nD) Təcrübə şərtlərinin bir hissəsini dəyişdirin\n\nCavab:" # old version
# #                "content": "Sual:\nŞagird təcrübəni təkrarlaya bilmək üçün ilk öncə hansı addımı atmalıdır?\nSeçimlər:\nA) Təcrübəni dəfələrlə aparmalıdır,\nB) bütün prosedurları qeydə almalıdır,\nC) Eksperimentin quruluşunun modelini hazırlamalıdır,\nD) Eksperimentin bəzi şərtlərini dəyişməlidir\n\nCavab:" # new arc
#             },
#             {
#                 "role": "assistant",
#                 "content": "B"
#             },
#             {
#                 "role": "user",
#                 "content": "Sual:\nDDT adlı kimyəvi maddə bir vaxtlar zərərli həşəratları öldürmək üçün istifadə olunurdu. Araşdırmalar bu kimyəvi maddənin bəzi quş növləri üçün zərərli olduğunu göstərdikdə, DDT-nin istifadəsi dayandırıldı. Elmi proses DDT-nin quşlara zərər verdiyini anlamalarına necə kömək edə bildi?\nSeçimlər:\nA) Kimyəvi maddə ilə bağlı müxtəlif proqnozlar verildi,\nB) Kimyəvi maddə ilə aparılan testlər fərqli nəticələr verdi,\nC) Kimyəvi maddə ilə təkrarlanan sınaqlar eyni nəticəyə gəldi,\nD) Kimyəvi maddənin xüsusiyyətlərini yoxlamaq üçün müxtəlif prosedurlardan istifadə edilmişdir\n\nCavab:" # old version
#                 # "content": "Sual:\nDDT adlı bir kimyəvi maddə bir zamanlar zərərverici həşəratları öldürmək üçün istifadə edilirdi. Araşdırmalar bu kimyəvi maddənin bəzi növ quşlar üçün zərərli olduğunu göstərdiyində, DDT-nin istifadəsi dayandırıldı. Elmi proses alimlərin DDT-nin quşlara zərərli olduğunu başa düşmələrinə ən yaxşı şəkildə necə kömək edə bildi?\nSeçimlər:\nA) Kimyəvi maddə haqqında müxtəlif proqnozlar verildi,\nB) Kimyəvi maddə üzərində aparılan testlər müxtəlif nəticələr verdi,\nC) Kimyəvi maddə üzrə təkrarlanan sınaqlar eyni nəticəni göstərdi,\nD) Kimyəvi maddənin xüsusiyyətlərini yoxlamaq üçün müxtəlif prosedurlar tətbiq edildi\n\nCavab:" # new version
#             },
#             {
#                 "role": "assistant",
#                 "content": "C"
#             },
#             {
#                 "role": "user",
#                 "content": "Sual:\nKainatdakı bütün qalaktikalar\nSeçimlər:\nA) çoxlu ulduzlardan ibarətdir,\nB) saatın istiqamətində dönür,\nC) Kimyəvi maddə ilə təkrarlanan sınaqlar eyni nəticəyə gəldi,\nD) eyni sayda ulduz var\n\nCavab:"
#                 # "content": "Sual:\nSəs dalğaları və okean dalğaları hər ikisi oxşardır çünki, hər ikisi\nSeçimlər:\nA) uzununa dalğalardır,\nB) eyni tezlikdədirlər,\nC) enerjini bir yerdən o biri yerə daşıyır,\nD) Müəyyən vasitəni bir yerdən başqa yerə aparır\n\nCavab:"
#             },
#             {
#                 "role": "assistant",
#                 "content": "A"
#                 # "content": "C"
#             },
        # #     {
        #         "role": "user",
        #         "content": f"Sual:\n{question}\nSeçimlər:\n{options}\n"
        #     }
        # ]


    elif dstype == 'qmc': # check it
        try:
            options = ",\n".join([f"{chr(65 + i)}) {option}" for i, option in enumerate(options)])
            print('OPTIONS:\n', options)
        except Exception as e:
            print(f"No options - Error: {e}")
            options = 'No options - Error'


        messages = [
            {
                "role": "system",
                "content": "You are an AI designed to answer questions in Azerbaijani based on the given article(context). Your task is to select the correct option from the given question and answer choices. You are given a question along with multiple options. Choose the correct option. Choose the single letter (A, B, C, D) that best answers the question. Respond with only the letter of the chosen answer, without any additional text."
            },
            {
                "role": "user",
                "content": 'Question:\nAzərbaycandan Gürcüstana hansı kitab kolleksiyaları göndərilib?\nContext:\nAzərbaycandan Gürcüstana kitab kolleksiyası göndərilib. Mədəniyyət və Turizm Nazirliyi tərəfindən Gürcüstan Milli Parlament Kitabxanasına, Tbilisi Dövlət Universitetinə, Marneuli şəhərində fəaliyyət göstərən Azərbaycan Mədəniyyət Mərkəzinə, Azərbaycan məktəblərinə, cəzaçəkmə müəssisələrinə xeyli sayda çap məhsulu göndərilib. Bunlar Azərbaycanın zəngin mədəniyyəti, tarixi, ədəbiyyatı, incəsənəti, müasir xarici siyasəti, Ermənistan-Azərbaycan Dağlıq Qarabağ münaqişəsi, eləcə də ümummilli lider Heydər Əliyev, prezident İlham Əliyev, Azərbaycanın birinci xanımı Mehriban Əliyeva haqqında kitab kolleksiyalarından, elektron nəşr və filmlərdir. Nazirlikdən verilən məlumata görə, kolleksiyaya "Heydər Əliyev və mədəniyyət", Elmira Axundovanın "Heydər Əliyev. Sözügedən kitabxanalara "Azərbaycan dilində latın qrafikası ilə kütləvi nəşrlərin həyata keçirilməsi haqqında" 2004-cü il 12 yanvar tarixli prezident sərəncamı əsasında nəfis şəkildə nəşr olunmuş Azərbaycan və dünya klassiklərinin əsərləri də göndərilib. Göndərilən kolleksiyalar əsasında Gürcüstan Milli Parlament Kitabxanasında Azərbaycan fondu və Azərbaycan guşəsi yaradılacaq.\nOptions:\nA) Heydər Əliyev və mədəniyyət,\nB) Elmira Axundovanın əsərləri,\nC) Azərbaycan mədəniyyəti haqqında kitablar,\nD) Müasir xarici siyasət kitabları\n\nAnswer:'
            },
            {
                "role": "assistant",
                "content": "A"
            },
            {
                "role": "user",
                "content": 'Question:\nGəncədə başlayan "Yeni Teatr" Respublika Festivalında hansı teatrlar iştirak edirlər?\nContext:\nGəncədə "Yeni Teatr" Respublika Festivalı başlayıb. Dünən Gəncədə ilk dəfə olaraq "Yeni Teatr" Respublika Festivalı start götürüb. Festivalda paytaxtdan Azərbaycan Dövlət Akademik Milli Dram, Gənc Tamaşaçılar, Bakı Bələdiyyə, Musiqili Komediya, "Yuğ", Pantomim teatrları qatılıblar. İştirakçılar sırasında Gəncə Dövlət Kukla Teatrı, Sumqayıt Dövlət Musiqili Dram, Gəncə, Qazax, Mingəçevir, Şəki, Lənkəran Dövlət Dram teatrları da var. Mədəniyyət və Turizm Nazirliyinin təşkil etdiyi bu festivalı teatr tənqidçiləri, tanınmış səhnə xadimləri də izləyərək qiymətləndirəcəklər. Festivalın açılışı Rafiq Səməndərin "Çətin məsələ" tamaşası ilə Gəncə Dövlət Kukla Teatrında gerçəkləşib.\nOptions:\nA) Bakı Bələdiyyə Teatrı,\nB) Naxçıvan Dövlət Teatrı,\nC) Abşeron Balet Teatrı,\nD) Şirvan Kukla Teatrı\n\nAnswer:'
            },
            {
                "role": "assistant",
                "content": "A"
            },
            {
                "role": "user",
                "content": "Question:\nQəşəm Nəcəfzadənin yeni kitabının adı nədir?\nContext:\nQəşəm Nəcəfzadənin yeni kitabı çapdan çıxıb. Tanınmış şair Qəşəm Nəcəfzadənin “Şirvannəşr” nəşriyyatında Qafqaz Universitetinin maliyyə dəstəyilə “Oğlum qapını aç, qapıda bir külək ölür” adlı şeirlər kitabı çap olunub. Kitabda şairin müxtəlif vaxtlarda və əsasən son illərdə yazdığı şeirləri toplanıb. Şeirlərin əsas məzmunu insan düyğularının təbiət və cəmiyyət, kainat və dünya hadisələri ilə əlaqəsindən, daha dəqiq insan bədii anatomiyasından, ölüm, həyat, müharibə, Allah, sevgi haqqında düşüncələrdən ibarətdir. Adından göründüyü kimi bu kitabda küləyin ölümü ədəbiyyatın, metoforanın, təbiətin, sözün ölümüdür. Bu ölüm, ölümlərin ən gözəli, həyatın isə başlanğıcıdır.\nOptions:\nA) Şirvannəşr,\nB) Qafqaz Universiteti,\nC) Oğlum qapını aç qapıda bir külək ölür,\nD) Küləyin ölümü\n\nAnswer:" 
            },
            {
                "role": "assistant",
                "content": "C"
            },
            {
                "role": "user",
                "content": "Question:\nBeynəlxalq Mədəniyyət günü hansı tarixdə qeyd olunur?\nContext:\nSəttar Bəhlulzadənin ev-muzeyində Beynəlxalq Mədəniyyət günü qeyd olunub. Səttar Bəhlulzadənin ev-muzeyində 15 aprel - Beynəlxalq Mədəniyyət gününə həsr edilmiş tədbir keçirilib. Tədbir Səttar Bəhlulzadənin ev-muzeyi və Əmircan qəsəbə Mədəniyyət evi ilə birgə təşkil olunub. Mədəniyyət evinin direktoru Afaq Bayramova mədəniyyət gününün qeyd olunma tarixi haqqında məlumat verib, Azərbaycan mədəniyyətinin dünya miqyasında təbliğ edilməsindən söhbət açıb. S.Bəhlulzadənin ev-muzeyinin baş mühafizi, rəssamın ailə üzvü, sənətşünaslıq üzrə fəlsəfə doktoru Pərzad Abdinova bugünkü tədbirin Xalq rəssamı Səttar Bəhlulzadənin evində keçirilməsinin təsadüfi olmadığını qeyd edərək, görkəmli rəssamın Azərbaycan və dünya mədəniyyəti xəzinəsini zənginləşdirən geniş yaradıcılığından danışıb. Tədbirdən sonra qonaqlar ev-muzeyinin ekspozisiyası ilə tanış olublar.\nOptions:\nA) 10 mart,\nB) 15 aprel,\nC) 21 iyun,\nD) 7 noyabr\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "B"
            },
            {
                "role": "user",
                "content": 'Question:\nFikrət Əmirovun əsəri hansı ölkənin paytaxtında səslənib?\nContext:\nFikrət Əmirov musiqisi Meksikada səslənib. Meksikanın paytaxtı Mexiko şəhərində Livan Respublikasının Mədəniyyət Günləri çərçivəsində silsilə konsert proqramları təşkil olunub. Mədəniyyət və Turizm Nazirliyinin dəstəyi ilə Mexiko şəhər filarmonik orkestrinin ifasında xalqımızın böyük bəstəkarı Fikrət Əmirovun "Ərəb mövzularında fortepiano və orkestr üçün konsert" adlı əsəri təqdim olunub. Mexiko Şəhər Teatrında keçirilən konsertdə orkestr tərəfindən həmçinin livanlı bəstəkar Camal Əbül Hüsn və Belçika əsilli Fransa bəstəkarı və orqanisti Sezar Frankın əsərləri də ifa olunub. Orkestri livanlı bədii rəhbər və baş dirijor Lubnan Baalbaki idarə edib. Fikrət Əmirovun musiqisi dərin məzmunu, zənginliyi, incəliyi, xəlqiliyi ilə dinləyiciləri özünə cəzb edib və alqışlarla qarşılanıb.\nOptions:\nA) Mərakeş,\nB) Mexiko,\nC) Madrid,\nD) Monako\n\nAnswer:'
            },
            {
                "role": "assistant",
                "content": "B"
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\nContext:\n{context}\nOptions:\n{options}\n"
            }
        ]


    elif dstype == 'kmc_biologiya': # Need to fix for each topic (10)  # mmlu-aze dataset
        try:
            cleaned_text = options.replace("' ", "',")

            # Convert the string to a Python list
            items = eval(cleaned_text)

            # Iterate over the list and format it as needed
            formatted_items = [f"{chr(65 + i)}) {item}" for i, item in enumerate(items)]

            # Join the formatted items into a string with commas
            options = ",\n".join(formatted_items)

            # Print formatted options
            print("OPTIONS:\n", options)

        except Exception as error:
            print(f"No options - Error: {error}")
            options = 'No options - Error'


        messages = [
            {
                "role": "system",
                "content": "You are an AI designed to answer questions in Azerbaijani based on biology. Your task is to select the correct option from the given question and answer choices. Choose the single letter (A, B, C, D, E) that best answers the question. Respond with only the letter of the chosen answer, without any additional text."
            },
            {
                "role": "user",
                "content": "Question:\n9 buğum varsa, neçə buğumarası var?\nOptions:\nA) 10,\nB) 9,\nC) 8,\nD) 4,\nE) 18\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "C"
            },
            {
                "role": "user",
                "content": "Question:\nAdi amöb nə ilə qidalanır?\nOptions:\nA) dəmirlə,\nB) balıqlarla,\nC) müalicəvi bitkilərlə,\nD) yosunlarla,\nE) quşlarla\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "D"
            },
            {
                "role": "user",
                "content": "Question:\nElektron mikroskoplar əşyanı necə dəfə boyudur?\nOptions:\nA) 270 dəfə,\nB) 100-270,\nC) 2-25 dəfə,\nD) 3600 dəfə,\nE) 10000-100000 dəfə\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "E"
            },
            {
                "role": "user",
                "content": "Question:\nInfuzor tərlik necə çoxalır?\nOptions:\nA) Konyuqasiya,\nB) Qeyri-cinsi,\nC) Sizoqaniya,\nD) Cinsi və qeyri-cinsi,\nE) Cinsi\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "D"
            },
            {
                "role": "user",
                "content": "Question:\nMürəkkəb çiçəklilər fasiləsinin formulunu qeyd edin\nOptions:\nA) O6 E6 D1,\nB) K5 L5 E5 D1,\nC) L6 E6 D1,\nD) K5 L4 E6 D1,\nE) K0 L5 E5 D1\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "E"
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\nOptions:\n{options}\n"
            }
        ]

    elif dstype == 'kmc_azerbaycan_tarixi': # Need to fix for each topic (10)  # mmlu-aze dataset
        try:
            cleaned_text = options.replace("' ", "',")

            # Convert the string to a Python list
            items = eval(cleaned_text)

            # Iterate over the list and format it as needed
            formatted_items = [f"{chr(65 + i)}) {item}" for i, item in enumerate(items)]

            # Join the formatted items into a string with commas
            options = ",\n".join(formatted_items)

            # Print formatted options
            print("OPTIONS:\n", options)

        except Exception as error:
            print(f"No options - Error: {error}")
            options = 'No options - Error'


        messages = [
            {
                "role": "system",
                "content": "You are an AI designed to answer questions in Azerbaijani based on historical facts. Your task is to select the correct option from the given question and answer choices. Choose the single letter (A, B, C, D, E) that best answers the question. Respond with only the letter of the chosen answer, without any additional text."
            },
            {
                "role": "user",
                "content": "Question:\nŞəddadilərə dövlətin paytaxtı hansı şəhər olmuşdur?\nOptions:\nA) Gəncə,\nB) Şəki,\nC) Şamaxı,\nD) Bərdə,\nE) Təbriz\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "A"
            },
            {
                "role": "user",
                "content": "Question:\nAzərbaycan Xalq Cümhuriyyətinin yaradılmasından düz ... ... sonra Sədərək-Diluci \"Umud\" körpüsü açıldı\nOptions:\nA) 72 il,\nB) 60 il,\nC) 74 il,\nD) 63 il,\nE) 70 il\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "C"
            },
            {
                "role": "user",
                "content": "Question:\nSənayeləşmə dövründə Azərbaycanda tikilib istifadəyə verilmiş müəssisələri müəyyən edin.\nOptions:\nA) Şmidt adına zavod,\nB) İlk krekinq qurğusu,\nC) Gəncə əyirici-toxucu kombinatı,\nD) Bakıda kondisioner zavodu,\nE) Xankəndində elektrik stansiyası,\nF) Qubada konserv zavodu\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "A"
            },
            {
                "role": "user",
                "content": "Question:\nAtropatın Midiya canişini təyin olunmasından neçə il sonra Atropatena müstəqil dövlətə çevrildi?\nOptions:\nA) 5 il sonra,\nB) 6 il sonra,\nC) 3 il sonra,\nD) 9 il sonra,\nE) 7 il sonra\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "E"
            },
            {
                "role": "user",
                "content": "Question:\nNeçənci ildə makedoniyalı İsgəndərin ölümündən sonra onun yaratdığı imperiya ayrı-ayrı dövlətlərə parçalandı?\nOptions:\nA) 304,\nB) 607,\nC) 323,\nD) 328,\nE) 542\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "C"
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\nOptions:\n{options}\n"
            }
        ]


    elif dstype == 'kmc_edebiyyat': # Need to fix for each topic (10)  # mmlu-aze dataset
        try:
            cleaned_text = options.replace("' ", "',")

            # Convert the string to a Python list
            items = eval(cleaned_text)

            # Iterate over the list and format it as needed
            formatted_items = [f"{chr(65 + i)}) {item}" for i, item in enumerate(items)]

            # Join the formatted items into a string with commas
            options = ",\n".join(formatted_items)

            # Print formatted options
            print("OPTIONS:\n", options)

        except Exception as error:
            print(f"No options - Error: {error}")
            options = 'No options - Error'

        messages = [
            {
                "role": "system",
                "content": "You are an AI designed to answer questions in Azerbaijani based on literary and historical facts. Your task is to select the correct option from the given question and answer choices. Choose the single letter (A, B, C, D, E) that best answers the question. Respond with only the letter of the chosen answer, without any additional text."
            },
            {
                "role": "user",
                "content": "Question:\nAlmas İldırım anadan olub:\nOptions:\nA) Xızıda,\nB) Çəmbərəkənddə,\nC) Gəncədə,\nD) Bakı şəhərində,\nE) Naxçıvanda\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "B"
            },
            {
                "role": "user",
                "content": "Question:\nCəfər Cabbarlı neçənci ildə anadan olmuşdur?\nOptions:\nA) 1901,\nB) 1934,\nC) 1899,\nD) 1900,\nE) 1903\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "C"
            },
            {
                "role": "user",
                "content": "Question:\nNəğmələr neçə yerə bölünür?\nOptions:\nA) 3,\nB) 5,\nC) 4,\nD) 6,\nE) 2\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "B"
            },
            {
                "role": "user",
                "content": "Question:\nAzərbaycan ədəbiyyatında ilk dəfə \"Bahariyyədən\" istifadə edərək novatorluq etmiş sənətkar?\nOptions:\nA) Ruhi,\nB) Zəmiri,\nC) Ş.İ. Xətai,\nD) İ. Nəsimi,\nE) Q. Burhanəddin\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "C"
            },
            {
                "role": "user",
                "content": "Question:\nSüleyman Sanı Axundovun əsərləri hansılardır? 1. \"Dibat bəy\". 2. \"Lənkəran\". 3. \"Şeyx Saban\". 4. \"Qaraca qız\". 5. \"Nurəddin\". 6. \"Azad\". 7. \"Xəncər\"\nOptions:\nA) 3, 4, 5, 6,\nB) 1, 4, 5,\nC) 1, 2, 5, 7,\nD) 1, 4, 5, 2,\nE) 1, 7, 6, 2\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "B"
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\nOptions:\n{options}\n"
            }

        ]


    elif dstype == 'kmc_tarix': # Need to fix for each topic (10)  # mmlu-aze dataset
        try:
            cleaned_text = options.replace("' ", "',")

            # Convert the string to a Python list
            items = eval(cleaned_text)

            # Iterate over the list and format it as needed
            formatted_items = [f"{chr(65 + i)}) {item}" for i, item in enumerate(items)]

            # Join the formatted items into a string with commas
            options = ",\n".join(formatted_items)

            # Print formatted options
            print("OPTIONS:\n", options)

        except Exception as error:
            print(f"No options - Error: {error}")
            options = 'No options - Error'

        messages = [
            {
                "role": "system",
                "content": "You are an AI designed to answer questions in Azerbaijani based on historical and cultural facts. Your task is to select the correct option from the given question and answer choices. Choose the single letter (A, B, C, D, E) that best answers the question. Respond with only the letter of the chosen answer, without any additional text."
            },
            {
                "role": "user",
                "content": "Question:\nƏhəməni dövlətinin banisi kimdir?\nOptions:\nA) 2-ci Kir,\nB) 1-ci Dara,\nC) 2-ci Kambiz,\nD) 3-cü Dara,\nE) Qaumata\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "A"
            },
            {
                "role": "user",
                "content": "Question:\nUyğur sözünün mənası nədir?\nOptions:\nA) güclü,\nB) mərd,\nC) qorxmaz,\nD) müttəfiq uyusan,\nE) igid qoçaq\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "D"
            },
            {
                "role": "user",
                "content": "Question:\nNeçənci əsrdə puritan təriqəti geniş yayılmışdı?\nOptions:\nA) XVI,\nB) XV,\nC) XVIII,\nD) XVIII əsrin ikinci yarısı,\nE) XVII\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "E"
            },
            {
                "role": "user",
                "content": "Question:\nBöyük kanalın uzunluğu nə qədərdir?\nOptions:\nA) 1234 km,\nB) 1200 km,\nC) 2000 km,\nD) 1500 km,\nE) 1700 km\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "E"
            },
            {
                "role": "user",
                "content": "Question:\nBerlində silahlı üsyan neçənci ildə baş verdi?\nOptions:\nA) 1848-ci il,\nB) 1812-ci il,\nC) 1832-ci il,\nD) 1889-cu il,\nE) 1864-cü il\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "A"
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\nOptions:\n{options}\n"
            }

        ]




    elif dstype == 'kmc_mentiq': # Need to fix for each topic (10)  # mmlu-aze dataset
        try:
            cleaned_text = options.replace("' ", "',")

            # Convert the string to a Python list
            items = eval(cleaned_text)

            # Iterate over the list and format it as needed
            formatted_items = [f"{chr(65 + i)}) {item}" for i, item in enumerate(items)]

            # Join the formatted items into a string with commas
            options = ",\n".join(formatted_items)

            # Print formatted options
            print("OPTIONS:\n", options)

        except Exception as error:
            print(f"No options - Error: {error}")
            options = 'No options - Error'


        messages = [
            {
                "role": "system",
                "content": "You are an AI designed to answer questions in Azerbaijani based on logical reasoning and problem-solving. Your task is to select the correct option from the given question and answer choices. Choose the single letter (A, B, C, D, E) that best answers the question. Respond with only the letter of the chosen answer, without any additional text."
            },
            {
                "role": "user",
                "content": "Question:\nSual işarəsinin yerinə nə olmalıdır? 9, 1, 8, 3, 7, ?, 6, 7, 5, 9\nOptions:\nA) 6,\nB) 8,\nC) 7,\nD) 5,\nE) 4\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "D"
            },
            {
                "role": "user",
                "content": "Question:\nFutbol oyunu zamanı meydançada bir komandanın 11 üzvü çıxış edir. Oyun ərzində bir komandada 2 sahə oyunçusu (sahə oyunçusu - qapıçı olmayan oyunçuları adlandırırıq) qırmızı vərəqə alaraq meydandan xaric edildi. Meydançada qapıçılardan savayı cəmi neçə oyunçu qaldı?\nOptions:\nA) 20,\nB) 18,\nC) 19,\nD) 21,\nE) 22\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "B"
            },
            {
                "role": "user",
                "content": "Question:\nAğac əkiləndə boyu 1 m idi. Ağac hər il 1 m boy atır. 5 il sonra ağacın boyundan 2 m kəsildi. Ağacın boyu nə qədər oldu?\nOptions:\nA) 3 m,\nB) 2 m,\nC) 5 m,\nD) 6 m,\nE) 4 m\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "E"
            },
            {
                "role": "user",
                "content": "Question:\nAta oğuldan 28 yaş böyükdür. 12 ildən sonra atanın yaşı oğulun yaşından 3 dəfə çox olacaqsa, indi oğulun neçə yaşı var?\nOptions:\nA) 1,\nB) 11,\nC) 2,\nD) 5,\nE) 12\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "C"
            },
            {
                "role": "user",
                "content": "Question:\nQanunauyğunluğu pozan ədədi göstərin: 47, 58, 71, 79, 95, 99, 119\nOptions:\nA) 99,\nB) 79,\nC) 71,\nD) 58,\nE) 119\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "A"
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\nOptions:\n{options}\n"
            }

        ]




    elif dstype == 'kmc_kimya': # Need to fix for each topic (10)  # mmlu-aze dataset
        try:
            cleaned_text = options.replace("' ", "',")

            # Convert the string to a Python list
            items = eval(cleaned_text)

            # Iterate over the list and format it as needed
            formatted_items = [f"{chr(65 + i)}) {item}" for i, item in enumerate(items)]

            # Join the formatted items into a string with commas
            options = ",\n".join(formatted_items)

            # Print formatted options
            print("OPTIONS:\n", options)

        except Exception as error:
            print(f"No options - Error: {error}")
            options = 'No options - Error'


        messages = [
            {
                "role": "system",
                "content": "You are an AI designed to answer questions in Azerbaijani based on chemistry and scientific concepts. Your task is to select the correct option from the given question and answer choices. Choose the single letter (A, B, C, D, E) that best answers the question. Respond with only the letter of the chosen answer, without any additional text."
            },
            {
                "role": "user",
                "content": "Question:\nHansı sabit 2 valentli elementdir?\nOptions:\nA) Li,\nB) Al,\nC) Ca,\nD) K,\nE) Na\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "C"
            },
            {
                "role": "user",
                "content": "Question:\n1 mol aren molekulu yanan zaman ayrılan suyu mol sayı n-3-ə bərabər olarsa aren molekulunda karbon atomlarının sayının müəyyən edin.\nOptions:\nA) n+6,\nB) n,\nC) n-3,\nD) n-2,\nE) 2n-6\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "B"
            },
            {
                "role": "user",
                "content": "Question:\n16q X birləşməsi 200 ml məhlulda 2M məhlul əmələ gətirir. X birləşməsinin nisbi molekul kütləsini hesablayın.\nOptions:\nA) 40,\nB) 74,\nC) 174,\nD) 142,\nE) 56\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "A"
            },
            {
                "role": "user",
                "content": "Question:\n200 qr 30%-li məhlulu 20%-li məhlula çevirmək üçün onun üzərinə neçə qram su əlavə etmək lazımdır?\nOptions:\nA) 140 q,\nB) 200 q,\nC) 300 q,\nD) 400 q,\nE) 100 q\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "E"
            },
            {
                "role": "user",
                "content": "Question:\nNeytral atom elektron alaraq: 1. Aniona çevrilir 2. Radiusu artır 3. Oksidləşir. Hansı ifadə doğrudur?\nOptions:\nA) yalnız 2,\nB) 1,2,\nC) yalnız 3,\nD) 2,3,\nE) yalnız 1\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "B"
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\nOptions:\n{options}\n"
            }

        ]



    elif dstype == 'kmc_azerbaycan_dili': # Need to fix for each topic (10)  # mmlu-aze dataset
        try:
            cleaned_text = options.replace("' ", "',")

            # Convert the string to a Python list
            items = eval(cleaned_text)

            # Iterate over the list and format it as needed
            formatted_items = [f"{chr(65 + i)}) {item}" for i, item in enumerate(items)]

            # Join the formatted items into a string with commas
            options = ",\n".join(formatted_items)

            # Print formatted options
            print("OPTIONS:\n", options)

        except Exception as error:
            print(f"No options - Error: {error}")
            options = 'No options - Error'

        messages = [
            {
                "role": "system",
                "content": "You are an AI designed to answer questions in Azerbaijani based on grammatical concepts and linguistics. Your task is to select the correct option from the given question and answer choices. Choose the single letter (A, B, C, D, E) that best answers the question. Respond with only the letter of the chosen answer, without any additional text."
            },
            {
                "role": "user",
                "content": "Question:\nDüzəltmə fellər neçə yollarla yaranır?\nOptions:\nA) 6,\nB) 4,\nC) 2,\nD) 5,\nE) 3\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "C"
            },
            {
                "role": "user",
                "content": "Question:\nBunlardan biri felin keçmiş zamanının şəkilçisidir:\nOptions:\nA) -malı,\nB) -mış,\nC) -ır,\nD) -ca,\nE) -ar\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "B"
            },
            {
                "role": "user",
                "content": "Question:\nSifətin neçə müqayisə dərəcəsi var?\nOptions:\nA) 3,\nB) 9,\nC) 5,\nD) 2,\nE) 7\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "A"
            },
            {
                "role": "user",
                "content": "Question:\nHansı cingiltili samitin kar qarşılığı yoxdur?\nOptions:\nA) C,\nB) Z,\nC) Y,\nD) M,\nE) B\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "D"
            },
            {
                "role": "user",
                "content": "Question:\nFelin indiki zamanı hansı şəkilçilərlə düzəlir?\nOptions:\nA) Dı di du dü,\nB) ır ir ur ür,\nC) Mış miş muş müş,\nD) Acaq əcək,\nE) Ar ər\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "B"
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\nOptions:\n{options}\n"
            }

        ]



    elif dstype == 'kmc_cografiya': # Need to fix for each topic (10)  # mmlu-aze dataset
        try:
            cleaned_text = options.replace("' ", "',")

            # Convert the string to a Python list
            items = eval(cleaned_text)

            # Iterate over the list and format it as needed
            formatted_items = [f"{chr(65 + i)}) {item}" for i, item in enumerate(items)]

            # Join the formatted items into a string with commas
            options = ",\n".join(formatted_items)

            # Print formatted options
            print("OPTIONS:\n", options)

        except Exception as error:
            print(f"No options - Error: {error}")
            options = 'No options - Error'

        messages = [
                {
                    "role": "system",
                    "content": "You are an AI designed to answer questions in Azerbaijani based on geographical and environmental knowledge. Your task is to select the correct option from the given question and answer choices. Choose the single letter (A, B, C, D, E) that best answers the question. Respond with only the letter of the chosen answer, without any additional text."
                },
                {
                    "role": "user",
                    "content": "Question:\nComolunqma zirvəsinin hündürlüyü?\nOptions:\nA) 8797 m,\nB) 1324 m,\nC) 8848 m,\nD) 7623 m,\nE) 6543 m\n\nAnswer:"
                },
                {
                    "role": "assistant",
                    "content": "C"
                },
                {
                    "role": "user",
                    "content": "Question:\nSakit okeanın sahəsi:\nOptions:\nA) 180 mln km²,\nB) 80 mln km²,\nC) 199 mln km²,\nD) 75 mln km²,\nE) 95 mln km²\n\nAnswer:"
                },
                {
                    "role": "assistant",
                    "content": "A"
                },
                {
                    "role": "user",
                    "content": "Question:\nAşağıda verilmiş hansı miqyas Qlobusun kiçik miqyasına uyğun gəlir?\nOptions:\nA) 1: 83000000,\nB) 1: 90000000,\nC) 1: 30000000,\nD) 1: 50000000,\nE) 1: 1000000\n\nAnswer:"
                },
                {
                    "role": "assistant",
                    "content": "A"
                },
                {
                    "role": "user",
                    "content": "Question:\nHeyvan və bitkilərin yayılmasını öyrənən elm sahəsi hansıdır?\nOptions:\nA) hidrologiya,\nB) geomorfologiya,\nC) geologiya,\nD) toponimika,\nE) biocoğrafiya\n\nAnswer:"
                },
                {
                    "role": "assistant",
                    "content": "E"
                },
                {
                    "role": "user",
                    "content": "Question:\nAvstraliyanın müstəqil materik olması fikri kimə məxsusdur?\nOptions:\nA) R.Skott,\nB) C.Kuk,\nC) V.Yanson,\nD) R.Piri,\nE) A.Tasman\n\nAnswer:"
                },
                {
                    "role": "assistant",
                    "content": "B"
                },
                {
                    "role": "user",
                    "content": f"Question:\n{question}\nOptions:\n{options}\n"
                }

            ]



    elif dstype == 'kmc_informatika': # Need to fix for each topic (10)  # mmlu-aze dataset
        try:
            cleaned_text = options.replace("' ", "',")

            # Convert the string to a Python list
            items = eval(cleaned_text)

            # Iterate over the list and format it as needed
            formatted_items = [f"{chr(65 + i)}) {item}" for i, item in enumerate(items)]

            # Join the formatted items into a string with commas
            options = ",\n".join(formatted_items)

            # Print formatted options
            print("OPTIONS:\n", options)

        except Exception as error:
            print(f"No options - Error: {error}")
            options = 'No options - Error'

        messages = [
            {
                "role": "system",
                "content": "You are an AI designed to answer questions in Azerbaijani based on technology and computer science. Your task is to select the correct option from the given question and answer choices. Choose the single letter (A, B, C, D, E) that best answers the question. Respond with only the letter of the chosen answer, without any additional text."
            },
            {
                "role": "user",
                "content": "Question:\nEstoniyanın internet domeni hansıdır?\nOptions:\nA) .ee,\nB) .lt,\nC) .eo,\nD) .es,\nE) .se\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "A"
            },
            {
                "role": "user",
                "content": "Question:\nGoogle neçənci ildə yaradıldı?\nOptions:\nA) 1997,\nB) 1985,\nC) 1999,\nD) 1698,\nE) 1969\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "A"
            },
            {
                "role": "user",
                "content": "Question:\nPublisher proqramında nəşr məhsuluna yeni səhifə hansı klaviş kombinasiyalarıyla yaradılır?\nOptions:\nA) Shift+Alt+N,\nB) Ctrl+A,\nC) Ctrl+Shift+N,\nD) Ctrl+Shift+M,\nE) Ctrl+N\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "E"
            },
            {
                "role": "user",
                "content": "Question:\nSketchUp proqramında bütün modellər ......... düzəlir?\nOptions:\nA) Tillərdən,\nB) heç biri,\nC) Düz xətlərdən,\nD) Nöqtələrdən,\nE) Tillər və üzlərdən\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "E"
            },
            {
                "role": "user",
                "content": "Question:\nİlk superkompüter kim tərəfindən yaradılıb?\nOptions:\nA) Çarlz Bebic,\nB) Vilhelm Leybnis,\nC) Seymur Krey,\nD) Blez Paskal,\nE) E. Post\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "C"
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\nOptions:\n{options}\n"
            }
        ]



    elif dstype == 'kmc_fizika': # Need to fix for each topic (10)  # mmlu-aze dataset
        try:
            cleaned_text = options.replace("' ", "',")

            # Convert the string to a Python list
            items = eval(cleaned_text)

            # Iterate over the list and format it as needed
            formatted_items = [f"{chr(65 + i)}) {item}" for i, item in enumerate(items)]

            # Join the formatted items into a string with commas
            options = ",\n".join(formatted_items)

            # Print formatted options
            print("OPTIONS:\n", options)

        except Exception as error:
            print(f"No options - Error: {error}")
            options = 'No options - Error'

        messages = [
            {
                "role": "system",
                "content": "You are an AI designed to answer questions in Azerbaijani based on physics concepts. Your task is to select the correct option from the given question and answer choices. Choose the single letter (A, B, C, D, E) that best answers the question. Respond with only the letter of the chosen answer, without any additional text."
            },
            {
                "role": "user",
                "content": "Question:\nHidravlik presdə porşenlər bərabər səviyyədə tarazladıqda, porşenlərin yerdəyişmələri nisbəti H2/H1=2 olarsa, porşenlərin altında mayelərin təzyiqləri nisbətini P2/P1 hesablayın.\nOptions:\nA) 4,\nB) 1/2,\nC) 3,\nD) 2,\nE) 1\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "A"
            },
            {
                "role": "user",
                "content": "Question:\n10 kg kütləli cisim 25 m/s² təcili nə qədər qüvvə ilə alır?\nOptions:\nA) 50 N,\nB) 25 N,\nC) 0.25 kN,\nD) 30 MN,\nE) 245 N\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "C"
            },
            {
                "role": "user",
                "content": "Question:\nSarğacdan axan cərəyanın şiddətini 0.3 A azaltdıqda sarğacın maqnit sahəsinin enerjisi 4 dəfə azaldı. Sarğacdan axan cərəyanın şiddətinin ilk qiymətini tapın.\nOptions:\nA) 1.8 A,\nB) 1.2 A,\nC) 0.6 A,\nD) 0.9 A,\nE) 0.8 A\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "C"
            },
            {
                "role": "user",
                "content": "Question:\nİşin görülmə yeyinliyini xarakterizə edən kəmiyyət necə adlanır?\nOptions:\nA) güc,\nB) yerdəyişmə,\nC) enerji,\nD) qüvvə,\nE) təcil\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "A"
            },
            {
                "role": "user",
                "content": "Question:\nKütləsi 80 kq olan idmançı xizəyin xəttində durmuşdur. Hər bir xizəyin uzunluğu 2 m, eni 10 sm olarsa, idmançının xizəyinin təzyiqi nə qədərdir?\nOptions:\nA) 34 Pa,\nB) 2.10 Pa,\nC) 12 Pa,\nD) 56 Pa,\nE) 7 Pa\n\nAnswer:"
            },
            {
                "role": "assistant",
                "content": "B"
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\nOptions:\n{options}\n"
            }
        ]


    # elif dstype == '' # fact_checker dataset  # need to create
        # try:
        #     options = ",\n".join([f"{chr(65 + i)}) {option}" for i, option in enumerate(options)]) 
        #     print('OPTIONS: ', options)
        # except Exception as e:
        #     print(f"No options - Error: {e}")
        #     options = 'No options - Error'


        # messages = [
        #     {
        #         "role": "system",
        #         "content": "You are an AI designed to answer questions in Azerbaijani based on {TOPIC NAME}. Your task is to select the correct option from the given question and answer choices. Choose the single letter (A, B, C, D) that best answers the question. Respond with only the letter of the chosen answer, without any additional text."
        #     },
        #     {
        #         # ..... # few-shots
        #     },
        #     {
        #         "role": "user",
        #         "content": f"Question:\n{question}\nOptions:\n{options}\n"
        #     }
        # ]


    else:
        raise("Invalid dstype")


    # generation_config = model.generation_config
    # generation_config.max_length = 10000
    # generation_config.pad_token_id = tokenizer.pad_token_id


    # # Dynamically set max_length to allow for input length + some buffer for output
    # input_length = encoding.input_ids.size(1)
    # max_length = input_length + 10000  # Add buffer for output generation


    def generate_answer(conversation, tokenizer, model):
        print("GENERATE ANSWER FROM HF MODEL")
        # Step 1: Prepare the conversation text using the tokenizer
        text_input = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        # print("\ntext_input:", text_input, '\n')

        # Step 2: Tokenize the input text
        encoding = tokenizer(text_input, return_tensors="pt").to("cpu")

        # Step 3: Generate the output from the model
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=encoding.input_ids,
                attention_mask=encoding.attention_mask,
                max_new_tokens=1000, 
                pad_token_id=tokenizer.pad_token_id,
            )

        # Step 4: Decode the output tokens into human-readable text
        # predicted_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted_answer = tokenizer.decode(outputs[0])
        print("\npredicted_answer_1 :", predicted_answer, '\n')

        # # Step 5: Extract answer (assuming the model is generating responses with "Answer:" prefix)
        # # We handle edge cases if the "Answer:" isn't found or there are unexpected formatting issues.
        # if 'Answer:' in predicted_answer:
        #     predicted_answer = predicted_answer.split('Answer:')[-1].strip()
        
        # # Optionally handle special tokens, like <|end_header_id|> or <|eot_id|>
        # # These tokens might not always appear, so we include some checks
        # predicted_answer = predicted_answer.replace('<|end_header_id|>', '').replace('<|eot_id|>', '')



        # REGULAR EXPRESSION VERSION:
        # Use regular expression to extract everything before the last occurrence of "<|eot_id|>"
        # text_before_last_eot = re.findall(r'(.*)<\|eot_id\|>', predicted_answer)[-1]

        # v2:(check it)
        # text_before_last_eot = re.findall(r'(.*)<\|(eot_id|im_end|finish|endoftext)\|>', predicted_answer)[-1]
        # text_before_last_eot = [match[0] for match in re.findall(r'(.*)<\|(eot_id|im_end|finish|endoftext)\|>', predicted_answer)][-1]


        # # Step 6: Print and return the final answer
        # print("\npredicted_answer_2 :", text_before_last_eot, '\n')
        # return text_before_last_eot

        # v3:
        # Extract only the part before the last `<|eot_id|>` tag and return the first part of the tuple
        match = re.findall(r'(.*)<\|(eot_id|im_end|finish|endoftext)\|>', predicted_answer)
        
        # Assuming you want the first part of the tuple (e.g., 'C') from the result
        if match:
            text_before_last_eot = match[-1][0]  # Get the first part (the answer letter)
        else:
            text_before_last_eot = ''  # No match found
        
        # Step 6: Print and return the final answer
        print("\npredicted_answer_2 :", text_before_last_eot, '\n')
        return text_before_last_eot



    def generate_answer_from_api(conversation, model):
        
        load_dotenv()
        
        OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY")

        # Initialize client for GPT 
        client_openai = OpenAI(
            # base_url=BASE_URL_LLM,
            api_key=OPENAI_API_KEY,
            # http_client=httpx_client
        )

        response = client_openai.chat.completions.create(
            model=model,
            messages=conversation,
            # max_tokens=500,  
            temperature=0,  
        )

        answer = response.choices[0].message.content

        return answer


    def generate_answer_from_gguf(messages, model, repo_id):
        from llama_cpp import Llama
        # import atexit

        def main():
            llm = Llama.from_pretrained(
                # repo_id="gaianet/Nemotron-Mini-4B-Instruct-GGUF",
                # filename="Nemotron-Mini-4B-Instruct-Q2_K.gguf",
                # # filename="Nemotron-Mini-4B-Instruct-Q4_0.gguf",

                repo_id = repo_id,
                filename = model,
                n_ctx=2048,
            )

            res = llm.create_chat_completion(
                messages = messages,
                temperature=0,
                # max_tokens=300,
            )

             
            # Extract the content from the response (checking that it's available)
            try:
                content = res['choices'][0]['message']['content']
                print("content:", content)
    
                print("GGUF Answer:", content)
            except KeyError:
                print("Error: Unable to extract content from the response.")

            print(f"repoId: {repo_id} and model: {model}")
            return content

        return main()



    # return generate_answer(messages, tokenizer, model) if not api else  generate_answer_from_api(messages, model) # for api and hf version
    # return main().generate_answer_from_gguf(messages, model, repo_id) if gguf else generate_answer(messages, tokenizer, model) if not api else generate_answer_from_api(messages, model) # for api and hf and gguf versions

    # Return v3: # for api and hf and gguf versions 
    if gguf:
        print("Generating answer using GGUF...")
        return generate_answer_from_gguf(messages, model, repo_id)
    elif api:
        print("Generating answer using API...")
        return generate_answer_from_api(messages, model)
    else:
        print("Generating answer using HF Tokenizer...")
        return generate_answer(messages, tokenizer, model)






def compare_answers(actual_answer: str, predicted_answer: str) -> int:
    """
    Compare the actual answer with the predicted answer.
    
    Parameters:
    - actual_answer (str): The correct answer.
    - predicted_answer (str): The answer predicted by the model.
    
    Returns:
    - int: 1 if the answers match, otherwise 0.
    """
    
    # return 1 if actual_answer.lower() == predicted_answer.lower() else 0 # v1
    print('\n\n')
    print("actual_answer:", actual_answer)
    print("predicted_answer:", predicted_answer)
    

    if pd.notna(predicted_answer) and isinstance(predicted_answer, str) and predicted_answer.strip():
        matched_predicted = re.match(r'[^A-Z]*([A-Z])', predicted_answer) # v2
    else:
        return 0

    if predicted_answer.lower() == "answer":
        return 0

    if matched_predicted and matched_predicted.group(1) is not None and matched_predicted.group(1) != "":
        print("matched_predicted:", matched_predicted.group(1).lower())
        print('\n\n')
    
        return 1 if actual_answer.lower() == matched_predicted.group(1).lower() else 0
    else:
        return 0










