"""Prompt engineering module optimized for DeepSeek reasoning models (deepseek-v4-flash).

DeepSeek-specific calibration:
- Persona-first immersion to override default AI assistant formality
- Positive framing ("write like X") over exhaustive negative lists
- Focused category-specific vocabulary injection (only relevant product vocab per call)
- Consolidated guardrails instead of 17 separate NEVER rules
- Real Daraz.pk training anchors presented as clear learning examples
- Structured JSON output with zero markdown fences or preambles
"""

from __future__ import annotations

import random

# ---------------------------------------------------------------------------
# Module-level Constants
# ---------------------------------------------------------------------------

INSTALLATION_KEYWORDS: tuple[str, ...] = (
    "air conditioner",
    "inverter ac",
    "split ac",
    "floor standing",
    "geyser",
)

MAJOR_APPLIANCE_BRANDS: tuple[str, ...] = (
    "haier",
    "dawlance",
    "gree",
    "tcl",
    "pel",
    "orient",
    "kenwood",
)

# ---------------------------------------------------------------------------
# Category Detection — maps product title keywords to focused vocab banks
# ---------------------------------------------------------------------------

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "refrigerator": ["refrigerator", "fridge", "freezer", "deep freezer", "hr-", "hrf-"],
    "dishwasher": ["dishwasher", "dish washer", "ddw-"],
    "washing_machine": ["washing machine", "clothes washer", "semi automatic", "twin tub", "hw-", "spin dryer", "spinner machine", "dryer machine"],
    "ac": ["air conditioner", "inverter ac", "split ac", "floor standing", "ton ac", "hsu-"],
    "blender": ["blender", "chopper", "grinder", "juicer", "spinner juicer", "citrus juicer", "food processor", "kitchen chef", "mixer", "hand blender"],
    "roti_maker": ["roti maker", "chapati maker"],
    "air_fryer": ["air fryer"],
    "toaster": ["toaster", "toast"],
    "iron": ["iron", "steam iron", "dry iron", "garment steamer"],
    "kettle": ["electric kettle", "kettle"],
    "microwave": ["microwave", "oven"],
    "water_dispenser": ["water dispenser", "dispenser"],
    "heater": ["heater", "fan heater", "room heater", "sun heater", "halogen heater", "quartz heater"],
    "straightener": ["straightener", "hair straightener", "styling tool", "flat iron", "straightening brush"],
    "hair_dryer": ["hair dryer", "blow dryer", "hair blow"],
    "curler": ["curler", "hair curler", "curling iron", "curling wand"],
    "trimmer": ["trimmer", "clipper", "shaver", "hair clipper"],
    "led_tv": ["led tv", "smart tv", "android tv", "google tv", "qled"],
    "humidifier": ["humidifier"],
    "geyser": ["geyser", "water heater", "instant geyser"],
    "pizza_pan": ["pizza pan", "pizza maker"],
    "sandwich_maker": ["sandwich maker", "panini"],
    "hot_plate": ["hot plate", "hotplate"],
    "scale": ["scale", "weight scale", "digital scale", "bath scale", "kitchen scale"],
    "insect_killer": ["insect killer"],
    "spray_mop": ["spray mop", "mop"],
    "fries_cutter": ["fries cutter", "slicer", "manual slicer"],
    "coffee_maker": ["coffee maker", "coffee machine", "espresso"],
    "cooler": ["air cooler", "room cooler", "evaporative cooler", "room air cooler"],
    "fan": ["table fan", "pedestal fan", "tower fan", "ceiling fan", "velocity fan", "bladeless fan"],
    "cooker": ["ceramic cooker", "induction cooker", "rice cooker"],
    "vacuum_cleaner": ["vacuum cleaner", "vacuum"],
    "air_purifier": ["air purifier"],
    "dehumidifier": ["dehumidifier"],
}

# Category-specific vocabulary banks — only the matching one gets injected
CATEGORY_VOCAB: dict[str, str] = {
    "refrigerator": (
        "Real buyers say: cooling bht fit hy bilkul chilled kar deta ha, compressor ki awaz bilkul nahi, "
        "ammi boht khush hoi, packing bht safe thi tut na jaye dar tha, color same aya, "
        "cooling best hay, warranty card stamped mila. "
        "4-star complaints: side py halka sa scratch delivery k waqt, handles thore nazuk lag rahe, "
        "wire b choti hy, delivery thori late hoye. "
        "3-star: chalta theek hy par body thori halki lagi guzara ha, cooling normal ha koi itni khaas nai."
    ),
    "washing_machine": (
        "Real buyers say: kapray bilkul saaf dhoti hy, spin speed boht fast 5 mint me almost sookh, "
        "time ki boht bachat h, ammi bht khush hain, awaz bilkul shor nahi karti, "
        "direct tap water se connect ho jati hai. "
        "4-star complaints: drain pipe thora chota alag se lgana para, water pressure high hona chahiye "
        "wrna error de deti, body plastic ki hai ehtiyat se use krna. "
        "3-star: spin k doran thora hilte vibrate krti, motor garam ho jati lagatar chalayein, guzara hy."
    ),
    "ac": (
        "Real buyers say: cooling achi hay paisa key hisab sey achi cheez miley, bijli kam khata meter slow chalta, "
        "bill me wazeh farq aya hy 4 ampere py chal raha, chilled cooling heating feature b check kia, "
        "zero noise level, installation team jaldi agai. "
        "4-star complaints: outdoor unit thora awaz karta, bas box thora phata hua tha "
        "dhang se handle nai kia, remote k cell sath nahi bheje. "
        "3-star: cooling theek hy normal ha boht super chilled nahi krta, inverter board trip kr jata low voltage py."
    ),
    "blender": (
        "Real buyers say: milk shake bohat hi maze ka bnta hy baraf b achi crush krta, blades bht sharp "
        "masala b pis jata hy, ammi k liye lia tha unko bht psnd aya, motor ki awaz normal, "
        "jug plastic heavy duty, full paisa wasool. "
        "4-star complaints: thori plastic ki smell ati shuru me chalao to, pulp container thora chota "
        "jaldi bhar jata, jug ka lock thora tight zor lagana prta. "
        "3-star: motor jaldi garam ho jati 2 minute se ziada chalao, blades itne sharp nahi tukray reh jatay."
    ),
    "roti_maker": (
        "Real buyers say: practice chahye pehle din kharab ab boht narm phooli hui roti banti hy, "
        "subah bacho k lunch k liye jaldi ban jati, non stick coating achi atta nahi chipakta, "
        "hostel walo k liye best hay bahir ki roti se jan chooti, plate garam jaldi hoti hai. "
        "4-star complaints: kinare thode kache reh jate hain haath se dabana parta, cord boht choti "
        "extension board use krna prta, aata narm goondhna parta wrna roti papad ban jati. "
        "3-star: itna asaan nai jesa video me dikhate hain, handle plastic ka bht halka toot jaye ga."
    ),
    "air_fryer": (
        "Real buyers say: bina oil k crispy fries bante hain healthy diet walo k liye, chicken tikka, overall nice product good experience, "
        "20 mint me juicy bake, basket size kafi bara touch panel smooth, chicken wings crispy oil free, "
        "paisa wasool cooking mey asani ki hay. "
        "4-star complaints: capacity thori kam bari family k liye 2 bar chalana prta, recipe book thori mushkil hay samagna, "
        "honi chahiye thi, shuru k 2 din plastic burning smell ab theek. "
        "3-star: fries crunch nahi hote jese oil me dry ban jate, non stick coating pehli wash me scratch."
    ),
    "toaster": (
        "Real buyers say: bread bilkul even brown banti hay, acha toaster hay overall nice product, quality wise sahi hay, "
        "perfect toast, chota compact kitchen counter pe jagah kam, crumb tray nikal k saaf karna asan, "
        "roz subah time bachta hy. "
        "4-star complaints: bara bread slice pura andar nahi ata bahir reh jata, level 4 se ooper "
        "bread jal jati setting sensitive, body thori garam ho jati bahir se. "
        "3-star: lever thora tight zor lagta plastic halka, aik side ziada brown dosri kam uneven heating."
    ),
    "iron": (
        "Real buyers say: bohat achi istri hay kaprey sahi iron karti hay, "
        "smooth gliding teflon base kapre jalte nahi, steam powerful wrinkles aik second me gayab, "
        "cord 360 ghoomti ulajhti nahi, thermostat accurate cut off karta. "
        "4-star complaints: steam me pani leakage ka halka dar spray button hard, taar ki length "
        "choti board pass hona chahiye, packaging thori dabbi hui thi, bas bijli thori zyada leta hay. "
        "3-star: steam tanki bht choti bar bar refill, maximum heat pe kapray chipakte, weight normal purani baat nahi."
    ),
    "kettle": (
        "Real buyers say: achi product hay galdi sey kam hogata hay bena gas key, steel body "
        "mazboot auto shut off perfect dry burn ka darr nahi, hostel k room maggie chai mints me, "
        "pure stainless steel no plastic smell boil hotay hi click off. "
        "4-star complaints: wire base thori loose table pe sahi rkhna prta, bahir se steel body bht "
        "garam hath lagane se bachein, handle ka plastic thora rough. "
        "3-star: 2 haftay me bottom pe white spots scale jam gaye, auto cutoff kbi kaam krta kbi nahi dar lagta."
    ),
    "microwave": (
        "Real buyers say: salan garam krne k liye best, useful product, buttons controls "
        "simple ammi asani se use kar leti, defrost zabardast gosht wagera sab best hay, "
        "glass turn table smooth ghoomti no extra noise. "
        "4-star complaints: andar ki light thori dim bahir se khana nahi dikhta, power cord choti, "
        "door push button thora hard dabana parta. "
        "3-star: heating thori slow pehle wale k mukable time double, body sheet metal patla dabta hy, guzara."
    ),
    "water_dispenser": (
        "Real buyers say: thanda pani jaldi chilled compressor bilkul silent, glass door look luxury "
        "taps soft leakage zero, refrigerator cabinet niche fruits cold drinks chilled, "
        "child safety lock garam tap pe, garmiyo k liye blessing. "
        "4-star complaints: bottle collar plastic thora loose lagta, compressor on hotay waqt thora "
        "vibrate halka sound, delivery 2 din late baki overall theek. "
        "3-star: cold water itna thanda nahi jitna purane me hota tha, mini fridge cooling nahi sirf naam ka cabinet."
    ),
    "heater": (
        "Real buyers say: 2 rods on karo pura kamra warm gas ki kami me best, tip-over switch safe "
        "giray to khud band, chota portable aik kamre se dosre le jana asan, red glowing light pyari, "
        "paisa wasool low electricity consumption. "
        "4-star complaints: body ka plastic halka heat se smell pehle ghante, taar moti honi chahiye "
        "plug warm ho jata, baray hall me asar kam small room k liye. "
        "3-star: room band kar k ghanta chalao tab farq, aik rod doosre din fuse ho gai, guzara hy."
    ),
    "straightener": (
        "Real buyers say: curly baal aik swipe me pin straight smooth finish, salon jesa look ghar pe "
        "temperature control buttons ache, heat 30 seconds me full ceramic plates baal nahi kheenchtin, "
        "cord 360 rotate styling easy, hair silk shiny jalne ki smell nahi. "
        "4-star complaints: plates thori choti ghane baalon k liye time ziada, bahir se outer body "
        "thori garam hoti, heat pouch sath nahi bheja travel issue. "
        "3-star: frizzy baal theek se straight nahi heat maximum pe kam, plates me gap press krna prta finish khas nahi."
    ),
    "hair_dryer": (
        "Real buyers say: powerful air throw 5 mint me baal dry, cold shot button zabardast, "
        "light weight hath thakta nahi blow dry k waqt, winter me subah office jane se pehle life saver, "
        "speed settings dono smooth nozzle best results. "
        "4-star complaints: taar thori choti mirror se plug door extension lagani, burning plastic smell "
        "high speed pe, packing carton thora damage delivery boy ki wajah se. "
        "3-star: awaz boht ziada jese jet plane ho heat control mushkil, body plastic halki handle garam build local."
    ),
    "curler": (
        "Real buyers say: beach waves 10 minute me pura head curl, barrel smooth baal ulajhte nahi "
        "curls sham tak stay, ceramic coating achi heat up jaldi wedding function perfect look, "
        "clamp tight baalon pe grip, salon jesi tight curls temperature indicator work karta. "
        "4-star complaints: barrel size thora bara short hair pe curl mushkil, stands weak table pe "
        "balance nahi gir jata, styling spray k bina curls jaldi khul jate. "
        "3-star: curls zra der nahi rukte 1 ghante me straight, clamp loose baal slip hath jalne ka khatra."
    ),
    "trimmer": (
        "Real buyers say: zero cut bilkul clean skin pe lagta nahi, 1 charge pe 4 dafa beard aram se, "
        "blade sharp smooth baal kheenchy bina, usb charging power bank se b charge, "
        "overall good product. "
        "4-star complaints: charging indicator light color change nahi pta nai kab full, oil ki bottle "
        "packaging me leak ho gai, thora vibration ziada hath me. "
        "3-star: heavy beard pe phans k chalta baal khenchta, plastic combs halkay clip loose ho gya battery 20 mint bus."
    ),
    "led_tv": (
        "Real buyers say: colors boht vibrant 4k video smooth, Netflix YouTube lag free voice control "
        "Google assistant kamal, display zabardast sound bar zaroorat nahi speakers bass clean, "
        "panel borders thin slim smart look, wifi jaldi pakar leta screen mirroring fast. "
        "4-star complaints: overall good bas agar is me sound thori aur achi hoti to best, "
        "slow restart. "
        "3-star: viewing angles side se picture wash out white, sound tinny bilkul bass nahi speaker phat-te unchi awaz."
    ),
    "humidifier": (
        "Real buyers say: mist throw strong winter dry throat khansi k liye best, zyada colors led "
        "aesthetic look bedroom night lamp ka kaam, aroma essential oil pura kamra khushboo, "
        "chota portable desk pe fit aik refill me pura din, ultrasonic mist silent. "
        "4-star complaints: charging cable choti switch pass hona zaroori, tanki safai mushkil, "
        "room bara ho to mist ka asar kam small room k liye. "
        "3-star: table pe paani jama ho jata floor wet, button touch kbi do bar dabana plastic halki girne se crack."
    ),
    "geyser": (
        "Real buyers say: tap kholo kuch der me garam pani agata, low gas pressure pe auto ignite, "
        "copper pipes heavy quality thermostat accurate, winter savior cold water tension khatam, "
        "bijli gas dono bachat batteries 6 mahine tak bhi chal jati. "
        "4-star complaints: water pressure sensor thora sensitive pump on krna prta, installation kit "
        "pipes quality normal bahir se li, ignition cells box sath nahi aye market se. "
        "3-star: gas low ho to bar bar flame band pressure stabilizer, temperature knob regulate nai ya ubalta ya normal."
    ),
    "pizza_pan": (
        "Real buyers say: 15 mint me crust crispy cheese fully melt oven ki zaroorat nahi, non stick "
        "coating top class bilkul nahi chipakta, pizza k ilawa chapati omelette shandar, "
        "handle cool rehta pakadne safe, bacho k liye homemade pizza easy timer bell accurate. "
        "4-star complaints: lid dhakkan ka glass delicate ehtiyat krni, thermostat switch thora loose "
        "ghoomate waqt, cleaning me dhayan switch socket me pani na jaye. "
        "3-star: upar se cheese brown nahi niche crust jal jati timer ziada ho, non stick coating jaldi peel off."
    ),
    "sandwich_maker": (
        "Real buyers say: bread crisp triangle pockets perfect cut and seal, non stick plates cheese "
        "chipakti nahi tissue se clean, 2 minute me crispy golden brown breakfast life saver, "
        "indicator lights ready to cook batati, handle lock mazboot press pe nahi tootta. "
        "4-star complaints: plates fixed removable nahi dhona mushkil, standard large bread thora bahir "
        "nikal ata small size k liye, wire boht short board k pas rakh k banana. "
        "3-star: handle lock clip pehle hafte toot gaya mota sandwich press me, heating uneven aik brown dosra kacha."
    ),
    "hot_plate": (
        "Real buyers say: 1500w coil jaldi red hot chai salan mints me gas load shedding ka tod, "
        "heavy cast iron plate har bartan adjust, hostel walo k liye best cooking companion, "
        "temperature dial smooth, thermostat auto cut off over heat se bacha. "
        "4-star complaints: shuru me coil se dhuwan smell pehli dafa, plate thandi hone me time bache "
        "hath na lagayein, taar garam hoti full heat pe lamba use. "
        "3-star: bijli boht khata meter fast bhagta sirf emergency chai, spiral coil uneven bartan balance nahi."
    ),
    "scale": (
        "Real buyers say: 1 gram tak accurate baking recipes perfect, tare function bowl zero kar k "
        "ingredients measure, clear LCD screen reading hold rehti, glass scale body weight track, "
        "gym diet portion control asan ho gaya paisa wasool. "
        "4-star complaints: surface plastic slippery flat surface pe rakh k measure, display backlight "
        "nahi thora light me dekhna, tare button slow response aik second wait. "
        "3-star: har bar wazan alag floor seedha na ho sensor balance, battery drain jaldi 10 din me cell change."
    ),
    "insect_killer": (
        "Real buyers say: room me andhera karo machhar khud kheench k aate zapped, coil k dhuway se "
        "jan chooti rat ko neend sukoon se, rechargeable racket shock powerful makhi machhar dher, "
        "uv light bright grid safe bacho k hath nahi lagte, dengue season zaroori. "
        "4-star complaints: room me light on ho to asar kam andhera chahiye, charging cable choti "
        "switch board k sath latkana, chotay insects grid me phans jate brush se nikalna. "
        "3-star: chotay fly nikal jate grid gap ziada, uv tube aik maheene me fuse market se nayi dhoondna mushkil."
    ),
    "spray_mop": (
        "Real buyers say: balti uthane ki zillat se jan choot gai spray me floor chamak, microfiber "
        "pad mota absorbent matti baal sab pakar leta, dettol pani dalo khushboo safai dono, "
        "trigger smooth mist broad tile dry jaldi, pocha lagana thakawat nahi. "
        "4-star complaints: bottle capacity 350ml kam bare lounge 2 bar bharni, rod fitting loose "
        "screw tight krna para, extra microfiber pad aik hi mila do hone chahiye. "
        "3-star: trigger 2 haftay baad phansna shuru spray throw nahi, bottle socket se paani leak ulta karo."
    ),
    "fries_cutter": (
        "Real buyers say: aalu rakho handle dabao aik second me McDonald jesi fries, solid heavy metal "
        "body zor se bend nahi, bacho k liye roz fries banana asaan finger cuts darr nahi, "
        "do size grid blades clean wash asan, steel wala life time chalega. "
        "4-star complaints: bara aalu ho to pehle half cut krna direct pura fit nai, suction base "
        "counter pe grip nahi zoor lagao hilta, blades change hath bachana. "
        "3-star: zor boht lagana aalu phans jata beech me, blade ka danda pehle din bend soft sabzi theek aalu weak."
    ),
    "coffee_maker": (
        "Real buyers say: 15 bar pressure rich thick crema coffee shop jesa taste, steam frother milk "
        "foam creamy cappuccino lovers k liye, 2 minute me fresh espresso shot heating fast, "
        "portafilter heavy brass, roz cafe k 800 rupay bach jate homemade latte. "
        "4-star complaints: cup warmer plate itni garam nahi cup pehle rinse, steam wand short bare "
        "milk jug me deep nahi jati, pehle 2 bar water flush plastic wash. "
        "3-star: coffee lukewarm nahi piping garam pakistani taste microwave krna, vibration ziada cup hil jata."
    ),
    "cooler": (
        "Real buyers say: ice pack dalo AC jesi chilled hawa phenkta bijli kam, honeycomb pads thick "
        "cooling zabardast 60 liter tank pura din, ups pe chal jata load shedding ka tod, "
        "water pump silent wheels smooth move asan, plastic body shock proof current khatra nahi. "
        "4-star complaints: fan speed 3 pe motor shor tv ki awaz band, honeycomb pads se pehle 2 din "
        "ghaas smell ab theek, auto swing kabhi atak jata. "
        "3-star: barsaat humidity me bilkul kaam nahi room chip chip, water pump aik maheene baad band pads sukhe."
    ),
    "fan": (
        "Real buyers say: copper winding original hawa throw door tak motor garam nahi, pedestal speed 1 "
        "pe toofani hawa metal blades balance perfect no vibration, silently kaam karta, "
        "parts assembly 15 minute ready, height adjustment lever smooth. "
        "4-star complaints: speed control knob plastic loose ghoomate waqt, base weight halka fan full "
        "speed wobble, delivery carton side se phata hua tha. "
        "3-star: awaz boht ziada raat neend kharab, oscillation atak atak gears plastic side ghoom k ruk jata."
    ),
    "cooker": (
        "Real buyers say: touch buttons instantly respond 1 liter pani 2 mint boil, ceramic surface "
        "clean cloth maro chamak flat cookware best, gas cylinder se hazar darje behter sasta, "
        "timer set karo be-fikr, smart cooking presets daal chai fry auto temperature. "
        "4-star complaints: internal exhaust fan shor ziada jab tak on rahe, sirf induction friendly "
        "bartan accept plain pe E0 error, touch panel sensitive gila hath beeps. "
        "3-star: bartan center se hiley to heating band sensors over sensitive, glass kinare pe metal ring nahi chip hone ka khatra."
    ),
    "vacuum_cleaner": (
        "Real buyers say: suction power toofani carpet andar se barik dhool matti nikal leta, wet dry "
        "feature floor pani suck sofa cleaning asan, drum capacity 18 liter jhanjhat nahi, "
        "blower function computer grills dhool second me saaf, pipe accessories sab sath aye. "
        "4-star complaints: awaz boht ziada motor sound pura mohalla sunta, taar length choti har "
        "kamre me plug badalna, wheels carpet pe move phanste plain floor ok. "
        "3-star: suction normal barik matti filter se blow out, hose pipe halka kheencho kink fold hawa ruk jati."
    ),
    "air_purifier": (
        "Real buyers say: Lahore ki smog k liye lifesaver AQI 350 se 25 pe le aya, subah gale me "
        "kharash band naak nahi HEPA filter cigarette smoke clear, Mi Home app connect mobile se "
        "fan speed monitor, night mode silent zero sound fresh clean air. "
        "4-star complaints: replacement HEPA filter costly 6 mahine badalna, indicator light raat "
        "bright tape lagani, door band rakhna prta wrna sensor green nahi hota. "
        "3-star: khushboo feel nahi bus hawa phenkta jese normal fan, filter Lahore pollution 2 maheene me 40% expensive."
    ),
    "dehumidifier": (
        "Real buyers say: deewaro ki seelan ka hal kamre se roz kafi pani nichod, wardrob me "
        "fungus musty smell khatam breathing fresh, compressor auto off tank full flood protection, "
        "continuous drain pipe option, AC k sath cooling double chip-chipahat end. "
        "4-star complaints: room temperature 1-2 degree warm exhaust heat se, unit wazan heavy wheels "
        "smooth pr uthana mushkil, tank nikalte pani chalak jata design ehtiyat. "
        "3-star: small mini model slow 24 ghante adha cup pani, compressor fridge jesa humming light sleepers mushkil."
    ),
    "dishwasher": (
        "Real buyers say: bartan bilkul saaf aur chamak jate hain, grease aur oil saaf ho jata hy, "
        "time ki boht bachat hy, ammi boht khush hain, tablets achi use karein to glass shine karta hy. "
        "4-star complaints: installation me plumber ki zrurat pari, rinse aid sath nahi tha, "
        "dry hone me thora time lagta hy. "
        "3-star: bare bartan karahi adjust krna mushkil hy, tablets mehnge hain, guzara hy."
    ),
}

# Fallback generic vocab when no category matches
GENERIC_VOCAB: str = (
    "Talk about: product quality achi hai, packing safe thi, "
    "same as shown in picture, delivery time par mili, value for money."
)


def detect_category(product_name: str) -> str:
    """Detects the product category from the product title.

    Args:
        product_name: Full product title string.

    Returns:
        Category key string matching CATEGORY_KEYWORDS, or 'generic' if no match.
    """
    name_lower = product_name.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in name_lower for kw in keywords):
            return category
    return "generic"


def get_category_vocab(category: str) -> str:
    """Returns the focused vocabulary bank for a detected category.

    Args:
        category: Category key from detect_category().

    Returns:
        Category-specific vocabulary guidance string.
    """
    return CATEGORY_VOCAB.get(category, GENERIC_VOCAB)


# ---------------------------------------------------------------------------
# Component Grounding — which physical parts exist per product type
# ---------------------------------------------------------------------------

COMPONENT_GROUNDING: dict[str, str] = {
    "blender": "Electric motor appliance. Has sharp blades, jar with lock, motor. NO compressor, NO cooling.",
    "roti_maker": "Electric heating appliance. Has non-stick plates, heating element. NO motor, NO compressor.",
    "air_fryer": "Electric heating appliance. Has basket, heating element, timer. NO motor, NO compressor.",
    "toaster": "Electric heating appliance. Has bread slots, heating coils, browning dial. NO motor, NO compressor.",
    "iron": "Electric heating appliance. Has soleplate, temperature dial. NO motor, NO compressor, NO sound.",
    "kettle": "Electric heating appliance. Has heating element, auto cut-off. NO motor, NO compressor.",
    "microwave": "Electric appliance. Has magnetron, turntable. NO motor sound, NO compressor.",
    "refrigerator": "Electric cooling appliance. Has compressor, shelves, freezer section. Compressor can be silent or have low hum.",
    "washing_machine": "Electric motor appliance. Has wash tub, spin tub, motor. Motor can be loud on high.",
    "ac": "Electric cooling appliance. Has compressor, outdoor unit, indoor unit. Inverter models are silent.",
    "straightener": "Electric heating appliance. Has ceramic/titanium plates. NO motor, NO compressor. NEVER say handle gets hot.",
    "hair_dryer": "Electric motor appliance. Has fan, heating element, speed/heat settings, nozzle. NO plates, NO compressor.",
    "curler": "Electric heating appliance. Has heated barrel/rod, clip, temperature dial. NO motor, NO fan noise.",
    "trimmer": "Battery/USB powered grooming device. Has sharp blades. NEVER use cooling or heavy motor language.",
    "led_tv": "Electronic display device. Has screen, speakers, smart OS. NO motor, NO compressor.",
    "pizza_pan": "Electric heating appliance. Has non-stick surface, heating element. NO motor.",
    "sandwich_maker": "Electric heating appliance. Has non-stick plates. NO motor.",
    "hot_plate": "Electric heating appliance. Has heating surface. NO motor, NO compressor.",
    "spray_mop": "Pure manual tool. NO plug, NO motor, NO electricity.",
    "fries_cutter": "Pure manual tool. NO plug, NO motor, NO electricity.",
    "geyser": "Gas/Electric water heating appliance. Has ignition, water tank.",
    "water_dispenser": "Electric cooling/heating appliance. Has taps, cooling compressor.",
    "heater": "Electric heating appliance. Has fan, heating element. Can have fan noise.",
    "humidifier": "Electric appliance. Has water tank, mist nozzle. Runs quietly.",
    "scale": "Battery powered device. Has digital display. NO motor.",
    "insect_killer": "Electric device. Has UV light. Runs silently.",
    "coffee_maker": "Electric heating appliance. Has jug, filter basket, warming plate, on/off switch. NO motor, NO compressor.",
    "fan": "Electric motor appliance. Has blades/tower body, speed dial/remote, oscillation. NO compressor, NO cooling liquid.",
    "cooler": "Evaporative cooling appliance. Has water tank, pump, cooling pad, fan. NOT an AC — does NOT use compressor.",
    "cooker": "Electric heating appliance. Has ceramic/induction surface, knob/touch controls. NO motor.",
    "vacuum_cleaner": "Electric motor appliance. Has suction motor, pipe, brush heads, dust bag/canister. Motor can be loud.",
    "air_purifier": "Electric appliance. Has HEPA filter, fan. Runs quietly. NO cooling, NO heating.",
    "dehumidifier": "Electric appliance. Has water tank, compressor/desiccant. Runs quietly.",
    "dishwasher": "Electric kitchen cleaning appliance. Has spray arms, dish racks, detergent dispenser. NO clothes washing, NO spin dryer.",
}


# ---------------------------------------------------------------------------
# SYSTEM PROMPT — Optimized for DeepSeek Reasoning Models
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """<persona>
You are NOT an AI. You ARE these real people — become them completely. Each person has a DISTINCT typing personality:

- TIKTOK GIRL (22, Lahore): Types fast, uses "😍" or "💯" sometimes, starts mid-thought like "mashallah acha product hay. recommended", mixes English slang randomly, excited tone. Might say "literally" in English mid-sentence.
- WORKING WOMAN (35, Karachi): Compares to old product or parlour visits, calm measured tone, writes "waise" and "khair" naturally, never uses emojis. Types complete thoughts but short.
- PINDI UNCLE (45, Rawalpindi): Writes broken English because keyboard is English, makes spelling mistakes even in English, uses "OK" and "good" a lot, might write ALL CAPS for emphasis on one word.
- SILENT BHAI (young guy, Islamabad): Just rates and writes 4-8 words max. "boht acha hy bhai 👍" and moves on. Zero detail. Zero story.
- KITCHEN AUNTY (housewife, Faisalabad): Ordered for kitchen, mentions WHO she made food for (bacho, mehman, ghar wale), uses "mashallah" or "alhamdulillah" naturally, keeps it simple.
- DETAIL GUY (28, office worker, Multan): The one person who actually writes a proper review with 2-3 sentences. Mentions specific things — model number, weight, build quality. Rare persona.
- SARCASTIC REVIEWER (30, Karachi): Even when giving 5 stars, has dry humor — "ab ye kehna nahi pary ga k kaam nahi krta" or backhanded praise style.

You just unboxed your order from KiaChahiye.com. You're typing a quick review on your phone. You are NOT writing a product description. You are sending a WhatsApp-style message about what you bought.
<mobile_typing_reality>
THIS IS HOW REAL PAKISTANIS TYPE ON MOBILE (copy these exact frequency patterns):

Spelling variations with real buyer frequency distributions — distribute your reviews accordingly:
- "hai" frequency: 80% "hy", 10% "hai", 5% "hay", 3% "h" (bare consonant), 2% "haa"
- "bohat" frequency: 50% "boht", 25% "bht", 15% "bohot", 10% "bahout"
- "theek" frequency: 49% "theek", 39% "thek", 12% "thk"
- "ziada" frequency: 54% "ziada", 41% "zyada", 5% "zda"
- "thanks" frequency: 45% "thanks", 38% "shukriya", 17% "thnx"
- "acha": write as "acha", "achi", "achaa", "achy"
- "bhi": shortcut to "b" (very frequent in reviews), "bhi", or "bi"
- "ke/ki/ka": shortcut to "k", "ke", or "ki"
- "mein": write as "me", "mein", "mei", "m"
- "lekin": write as "lekin", "lekn", "lkn", "pr", "par"
- "karna/karni": write as "krna", "krni", "karna"
- Dropped terminal nasalization (standard informal Roman Urdu): "deewaro" not "deewaron", "garmiyo" not "garmiyon", "kapro" not "kapron", "baalo" not "baalon"

Real mobile habits:
- Run-on sentences with no periods: "product acha hy packing b achi thi delivery time pe hogai"
- Very few periods — most reviews have 0-1 periods total
- Some reviews are one long flowing sentence with no punctuation at all
- Dropped spaces sometimes: "hogya", "hogai", "hojata", "agya"
- Grammatical "mistakes" that are actually how people talk: "bohat acha product hai ye wala"
- Mix English product loanwords naturally into Urdu syntax: "heat jaldi ati hy baal smooth hojatay hain", "chilled kar deta ha", "compressor ki awaz bilkul b nahi hy"

Urdu filler words (sprinkle naturally — NOT in every review, only 1-2 per batch where appropriate):
- "waise" — btw/incidentally: "waise packing boht achi thi"
- "khair" — anyway: "khair chalta hy kaam chal rha hy"
- "aray" — exclamation: "aray boht zabardast hy"
- "matlab" — like/meaning: "matlab expected se better nikla"
- "haan" — yes/affirmation mid-sentence: "haan bhai original piece mila"
- "bus" — just/that's it: "bus itna kahunga ke paisa wasool"
- "woh b" — that too: "cooling achi hy woh b bilkul silent"
</mobile_typing_reality>

<voice_calibration>
STUDY THESE REAL KIACHAHIYE.COM / DARAZ.PK REVIEWS — your output must sound EXACTLY this raw and natural:

5-Star Real Reviews (notice the RANGE: some pure English, some pure Urdu, some mixed, some short, some medium):
- "very good value for money and recommended. Best price"
- "same as shown in the picture."
- "I am satisfied to this product. Too good and brand new product. Thanku KiaChahye"
- "good product came in secure packing. Definitely recommend this product"
- "Good quality. Reached within one day. Complete in original condition. Working perfectly."
- "Very nice product same as shown in pictures"
- "mashallah se bahout Ache machine haa or motor ki speed bhi tez haa thanks kiachahye"
- "Recommnded Price with free delivery and installation"
- "Quality aur customer services best thi."
- "Very spacious and beautiful fridge in very good price"
- "Overall a good product and looks all fine so far. Thanks kiachahye"
- "I am very happy with this product. Very beautiful model. Thanks kiachahye"
- "the fridge is excellent in terms of cooling and quality. received on time"
- "Fridge bilkul silent hai aur thanda boht tezi se karta hai."
- "Pehle semi-automatic thi, leken is se kaam boht asaan ho gaya. Motor power zabardast hai aur kapray bilkul saaf dho leti hai."
- "Achi quality hai, delivery ke sath installation bhi free karwaye. recommended"
- "I recived my order bohat hi zabardast hy same wesa hi hy jysy picture main tha thanks kiachahye"
- "boht zabardast fridge hai cooling bht fit hy bilkul chilled kar deta ha time pe delivery mili ammi boht khush hoi"
- "mashallah boht pyara fridge hy color b same aya awaz bilkul b nahi hy compressor ki 10/10 recommended"
- "10/10 machine spin speed boht fast hy 5 mint me kapre almost sookh jate hain highly recommended"
- "milk shake 1 minute me ban jata hy baraf b achi crush krta blades bht sharp hain ammi ko bht psnd aya"
- "salon jesa look ghar pe mil jata hy curly baal aik swipe me pin straight ceramic plates smooth hain"

NOTICE THE PATTERNS:
- English reviews have BROKEN grammar: "satisfied TO this", "Thanku", "Recommnded", "recived", "jysy" — real typos!
- Some reviews are ALL English with periods between sentences
- Some are ALL Roman Urdu with zero punctuation
- Some MIX both naturally: "mashallah se bahout Ache machine haa" uses both
- "hay" and "haa" appear alongside "hy" and "hai" — NOT just one spelling
- Platform thanks ("thanks kiachahye") appears naturally in some

4-Star Real Reviews (notice: complaint woven naturally with "bas"/"lekin"/"but"):
- "mashallah se bahout Ache machine haa bas pip thora chota hay"
- "Overall a good product and looks fine but noise sound hay but not too much"
- "Packaging was awesome product bhi bohat acha lag raha hay leken delivery late hoye"
- "product is really fantastic but the delivery was late. overall experience is pretty good."
- "Machine silent aur solid hai, bas tap connector pipe thora tight tha lagate waqt."
- "Iron kapray bilkul crisp press karta hai, heating teez hai. Bas water spray wala button thora hard press hota hai."
- "Juicer original hai aur kaam secondon mein karta hai, bas sound thori zyada hai high speed par. Overall value for money."
- "Is rate mein inverter fridge boht achi value hai. Bas thora sa bend hay grill mey"
- "Product theek chal rahi hai lekin delivery late mili."
- "Product sahi hay Bas power cord thori short hai, extension lagana para. Baqi kaam fit hai."
- "cooling achi hy freezer bara hy lekin handles thore nazuk lag rahe hain wire b choti hy"

3-Star Real Reviews (notice: pragmatic resignation / reluctant acceptance — 'guzara hy'):
- "chalta theek hy par body thori halki lagi pehle wale pel k muqable me plastic b normal hy guzara ha"
- "guzara hy motor garam ho jati hy agar 2 dafa lagatar chalayein cover b sath nai bheja seller ne"
- "guzara hy fries crunch nahi hote jese oil me hotay hain thode dry ban jate hain baking k liye behtar h"
- "heating thori slow hy pehle wale microwave k mukable me time double lagana prta hy average unit"

CRITICAL: Match the EXACT messiness level of these examples. Notice:
- "hay" and "haa" alongside "hy" — mix these up!
- "bas" is the #1 transition before complaints (not always "lekin")
- English reviews have REAL typos ("recived", "Recommnded", "pip") — not polished grammar
- Some reviews end with periods, some don't — BOTH are real
- "thanks kiachahye" appears naturally without feeling forced
</voice_calibration>

<anti_bot_checklist>
YOUR REVIEW SOUNDS LIKE A BOT IF ANY OF THESE ARE TRUE — rewrite it:
✗ Every sentence has correct grammar and proper punctuation
✗ You wrote "hai" the same way in every review (should mix hy/h/hai/hay/haa)
✗ Review reads like a product description: "heats up in seconds, hair feels soft after using"
✗ Over-explaining or writing backstories: on Daraz, most buyers just write 1 punchy line under 10 words
✗ All reviews in the batch have similar word count or structure
✗ Sentences follow [Feature] + [positive adjective] + period pattern
✗ You used proper conjunctions like "aur" at the start of every clause instead of just running words together
✗ Review has a neat, tidy ending like "Overall satisfied." or "Highly recommended." or "works perfectly"
✗ You capitalized words properly throughout
✗ English reviews have perfect grammar — real Pakistani English has typos and broken grammar
✗ Every 4-star review uses "lekin" — real buyers say "bas" or "but" just as often

FIX: Imagine you're lying in bed, phone in one hand, typing with your thumb about what you just received. THAT is the energy.
</anti_bot_checklist>

<product_rules>
PHYSICAL REALITY (never cross-contaminate between product types):
1. Only mention parts that physically exist on THIS product. Blades for blenders, plates for roti makers, compressor for fridges — NEVER mix them up.
2. Scale Truth: If title says 8+ Cu Ft fridge, it's a full family appliance — NEVER call it "small".
3. ZERO RAW NUMBERS: Real buyers NEVER quote specs. Say "motor tez hai" NOT "600W motor". Say "room ke liye perfect" NOT "1.5 ton".
4. FOOD WORDS: Say "khana", "salan", "roti", "chai". NEVER say "curry".
</product_rules>

<guardrails>
HARD RULES (violating any = failure):

1. BANNED WORDS:
   - Indian Hindi: "turant", "dhanyawad", "suvidha", "kripya", "upayog"
   - Formal Urdu: "mayari", "paidaar", "faraham"
   - AI buzzwords: "efficient", "optimal", "satisfactory", "seamless", "delighted", "craftsmanship", "game-changer", "sleek design", "pinnacle", "epitome", "boasts", "testament to"
   - Bookish words: "foran" / "fauran" — use "jaldi" instead

2. REALITY RULES:
   - Recent delivery only (1-2 weeks). NEVER claim months of use.
   - NEVER mention "bewi"/"biwi"/"wife" — use "ghar ke liye", "ghar walon ko pasand aya"
   - NEVER mention broken parts, fake items, sparking, chemical smell
   - NEVER show self-correction ("wife... nai" or "I mean")
   - NEVER write "pehle online order karne me dar tha" or similar online trust hesitation clichés — real buyers talk about using the item itself!

3. ANTI-REPETITION:
   - Every review: different structure, different opening, different angle
   - NEVER start multiple reviews with the same word
   - These phrases MAX ONCE per batch: "paisa wasool", "same as shown in picture", "recommended", "boht achi cheez hai", "maza agya"
   - "ammi ke liye" MAX once — rotate: "behen k liye", "apne liye", "office ke liye", "hostel k liye", "bacho k liye"
   - OPENING DIVERSITY: "ghar k liye" / "ghar k liye mangwaya" / "ghar k liye lia" MAX ONCE per batch! Other reviews must start differently — with the product action, a reaction, or mid-thought.
   - NEVER start 2+ reviews with "boht" or "very" or "good" in the same batch

4. ENDINGS — 70% NO ENDING RULE:
   - 70% of reviews MUST just STOP after the product comment. No sign-off, no closing word, nothing. Just end mid-thought like a real person.
     Example: "heat jaldi ati hy baal smooth hojatay hain" ← just stops here, no "recommended" or "overall good"
   - Only 30% may have a casual closer. Pick from: "recommended", "10/10", "worth it"
   - NEVER end with: "Overall satisfied.", "Highly recommended.", "Good product.", "works perfectly" — these are bot signatures

5. SPEAKER GENDER MATCH (Aurat ya Mard):
   - Match the reviewer's perspective and verbs to their assigned name:
     • FEMALE names (e.g. Ayesha, Fatima, Zainab, Sana, Hira, Sadia): She is an aurat (woman). Self-verbs: "khush hui", "use karti hoon", "apne liye mangwaya tha". For female styling (straightener, curler), she is the direct user.
     • MALE names (e.g. Bilal, Usman, Tariq, Ahmed, Farhan, Kamran): He is a mard (man). Self-verbs: "khush hua", "use karta hoon".
     • Cross-gender items: If a male name reviews female styling products (hair straightener, curler), he bought it for family: "sister k liye lia tha", "ghar walon k liye mangwaya", "ammi boht khush hain". If a female name reviews a men's trimmer: "bhai k liye lia tha" or "gift dia tha".

6. FORMATTING:
   - Correct Urdu gender for objects: "pizza perfect banta hai" (masc), "roti soft banti hai" (fem)
   - No trailing dots "....". Minimal punctuation — most reviews have 0-1 periods
   - 👍 emoji in at most 1 review per batch (not every batch)
</guardrails>

<review_tiers>
TIER 1 — SEEDHI BAAT (4-12 words): MOST COMMON ON DARAZ (50–60% of buyers). Quick, direct, one-thumb reaction. Real buyers just tap stars and write what matters:
  - "same as shown in the picture."
  - "very good value for money and recommended. Best price"
  - "good product came in secure packing. Definitely recommend this product"
  - "Good quality. Complete in original condition. Working perfectly."
  - "Very nice product same as shown in pictures"
  - "mashallah se bahout Ache machine haa or motor ki speed bhi tez haa"
  - "Fridge bilkul silent hai aur thanda boht tezi se karta hai."
  - "Quality aur customer services best thi."
  - "MashaAllah original piece mila 👍"

TIER 2 — PRACTICAL SHORT (12-25 words): SECONDARY TIER (35–45% of reviews). 1-2 run-on sentences with a specific observation or minor 4-star friction:
  - "Machine silent aur solid hai, bas tap connector pipe thora tight tha lagate waqt."
  - "Packaging was awesome product bhi bohat acha lag raha hay leken delivery late hoye"
  - "Iron kapray bilkul crisp press karta hai, heating teez hai. Bas water spray wala button thora hard press hota hai."
  - "Juicer original hai aur kaam secondon mein karta hai, bas sound thori zyada hai high speed par."
  - "chalaya aur pehli dafa me hi samajh agya koi rocket science nai motor tez hy blades sharp"
  - "Pehle semi-automatic thi, leken is se kaam boht asaan ho gaya. Motor power zabardast hai"
  - "Product sahi hay Bas power cord thori short hai, extension lagana para. Baqi kaam fit hai."

TIER 3 — DETAILED (28-40 words, RAREST — only when specifically allowed): 2-3 natural sentences about a real situation (replacing old appliance, weekend use). NEVER an essay:
  - "humara purana blender 5 saal baad kharab hua tha to ye wala lia motor ki speed us se kahin ziada tez hy baraf aur badam aik minute me powder bana deta hy jug heavy plastic ka hy"
  - "choti behen ki shadi k liye lia tha box open kr k pura unit test kia finish boht luxury lagti hy buttons responsive hain warranty card b andar stamped mila full satisfaction"
  - "sunday ko pehli dafa use kia pehle thora dhyan se instructions parhein phir fries banaye bina oil k bilkul crispy nikle bacho ko bht maza aya safai b tissue se asan ho gai"
</review_tiers>

<rating_sentiment>
5 Stars: Genuine reactions from different TYPES of people. NOT generic praise. Each review must feel like it was typed by a completely different human with a different personality, reason for buying, and thing they noticed first. Variety is EVERYTHING.

4 Stars: Happy with product BUT one small real complaint naturally woven into the review (not tacked on at the end). Pick a DIFFERENT friction angle for each 4-star review:
  A. TRANSIT: "box ka corner thora daba tha transit mein lekin product safe nikla"
  B. COURIER TIMING: "delivery thori late hogai thi" or "rider late aya lekin parcel safe tha"
  C. CORD/CABLE: "wire thora chota hy extension lagani pari"
  D. SOUND ON HIGH: "tez speed pe thori awaz ati hy"
  E. SIZE: "size socha tha us se thora bara nikla"
  F. WEIGHT: "thori bhari hy sochne se"
  G. COLOR: "colour thora different hy screen se lekin in person theek hy"
  H. ACCESSORIES: "extra attachment aata to aur acha hota"
  I. FIRST USE: "pehli baar chalaya to halki smell aayi ek din me chali gai"
  J. PACKING: "tape boht tight tha box kholne me mushkil hui"

3 Stars: Pragmatic resignation / reluctant acceptance ("guzara hy", "normal chez hy bus", "itna khas nai"). The device works or meets minimum bare function, but feels light/cheap, motor runs warm, heating is slow, or takes longer than expected. It's not completely broken, just mediocre value or slight disappointment ("chalta theek hy par body halki lagi guzara ha").

DELIVERY: ZERO or ONE review per batch mentions delivery. NEVER more than one. 80% of batches should have ZERO delivery mentions — real buyers talk about the PRODUCT, not delivery.
CUSTOMER SERVICE: Only mention if SPECIFICALLY instructed. Vary phrasing every time.
FREE INSTALLATION: Only if specifically instructed. Only in 5-star reviews. Keep it brief.
PLATFORM THANKS: At most 1 review in 10-15 products may write "thanks kiachahye". Don't force it.
</rating_sentiment>

<output_format>
Return STRICTLY a valid JSON array (starting with "[" and ending with "]").
Do NOT wrap in markdown codeblocks (no ```json or ```).
Do NOT include any introduction, explanations, or commentary outside the JSON array.
[
  {"name": "<Pakistani Name>", "rating": <int>, "review": "<review text>"}
]
</output_format>"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_installation_candidate(product_name: str) -> bool:
    """Checks if a product belongs to a major appliance category eligible for free installation."""
    name_lower = product_name.lower()
    return any(k in name_lower for k in INSTALLATION_KEYWORDS)


CATEGORY_ANGLES: dict[str, list[str]] = {
    "trimmer": [
        "Battery life, charging speed & cordless convenience",
        "Blade sharpness, skin comfort & zero tugging on thick hair",
        "Grip ergonomics, lightweight hand feel & travel portability",
        "Gift for family / brother / husband or personal grooming routine",
        "Unboxing condition, accessories & comb attachments quality",
        "Motor sound & vibration level during prolonged use",
    ],
    "ac": [
        "Cooling speed & room temperature drop in extreme heat",
        "Low electricity consumption & inverter meter ampere savings",
        "Quiet indoor unit operation & sleep comfort",
        "Delivery handling, outer box packaging & installation team",
        "Remote control functionality, heating/cooling mode toggle",
    ],
    "refrigerator": [
        "Quick ice making & cooling retention during power cuts",
        "Silent compressor operation & zero humming noise",
        "Spacious glass shelves, door pockets & vegetable crisper",
        "Exterior color finish, safe delivery packing without scratches",
        "Thermostat accuracy & electricity efficiency",
    ],
    "washing_machine": [
        "Washing agitation power on tough stains and collars",
        "Spin dryer speed and quick drying time",
        "Water drainage pipe connection and tap fitting",
        "Motor stability and minimal vibration on level floor",
        "Tub capacity and ease of daily laundry loads",
    ],
    "blender": [
        "Milkshake and fruit smoothie blending speed",
        "Spice and dry masala grinding fineness",
        "Jug plastic thickness and lid lock tightness",
        "Motor power and noise level under load",
        "Blade sharpness and ice crushing capability",
    ],
    "air_fryer": [
        "Crispy fries and snacks with zero or minimal oil",
        "Chicken wings and tikka baking time and juiciness",
        "Basket capacity for family meals and non-stick coating",
        "Touch panel presets and temperature control ease",
        "Cleaning convenience and odor-free kitchen cooking",
    ],
    "toaster": [
        "Even browning level and toast crispiness",
        "Crumb tray removal and ease of cleaning",
        "Compact size on kitchen counter",
        "Bread slice thickness fitting in slots",
    ],
    "iron": [
        "Teflon base smooth gliding and zero fabric sticking",
        "Steam blast wrinkle removal on cotton and linen",
        "Cord rotation flexibility and reach to plug",
        "Thermostat temperature cutoff accuracy",
    ],
    "kettle": [
        "Boiling speed for tea/coffee under 2-3 minutes",
        "Stainless steel body durability and auto cutoff safety",
        "Hostel or office desk tea preparation naimat",
        "Handle heat insulation and pouring spout design",
    ],
    "microwave": [
        "Food reheating speed and tea warming in 30 seconds",
        "Meat defrosting uniformity and turntable rotation",
        "Simple button controls for family members/parents",
        "Internal capacity and plate fitting",
    ],
    "generic": [
        "Initial unboxing, packaging safety & visual finish",
        "Practical performance & daily household convenience",
        "Durability feel, button/knob quality & build materials",
        "Value for money compared to market alternatives",
        "Ease of cleaning, storage & portability",
    ],
}


def build_review_prompt(
    product_name: str,
    sku: str,
    ratings: list[int],
    allow_detailed: bool = False,
    include_installation: bool = False,
    include_customer_service: bool = False,
    recent_reviews: list[str] | None = None,
    rejected_reviews: list[str] | None = None,
) -> str:
    """Builds the structured user prompt for review generation with anti-repetition memory.

    Args:
        product_name: The full product title as listed on the store.
        sku: SKU or model identifier (used as context, not in review text).
        ratings: Target star ratings list (e.g. [5, 5, 4, 5, 4]).
        allow_detailed: If True, allows one Tier 3 detailed review in the batch (~20% of products).
        include_installation: If True, instructs LLM to include one free installation mention
            (~35% of eligible appliances — controlled probabilistically in llm_client.py).
        include_customer_service: If True, instructs LLM to include one customer service mention
            (~25% of products across catalog).
        recent_reviews: Optional list of recent review texts to avoid repeating.
        rejected_reviews: Optional list of reviews discarded in current session to avoid regenerating.

    Returns:
        Structured user prompt ready to send as the 'user' message to the LLM.
    """
    from persona_data import generate_reviewer

    clean_sku = sku.strip() if sku else "N/A"
    count = len(ratings)
    # Tier 1 (Seedhi Baat) is strictly dominant (60%+ on Daraz): e.g. 5 reviews -> 3 Tier 1, 2 Tier 2
    # Tier 2 (Practical Short) is secondary: 1-2 reviews
    # Tier 3 (Detailed) is rarest: at most 1 review when allow_detailed is True
    if count <= 2:
        seedhi_baat_count = 1
        tier2_count = count - seedhi_baat_count
        eng_count = "1"
    elif count == 3:
        if allow_detailed:
            seedhi_baat_count = 1
            tier2_count = 1
        else:
            seedhi_baat_count = 2
            tier2_count = 1
        eng_count = "1"
    elif count == 4:
        if allow_detailed:
            seedhi_baat_count = 2
            tier2_count = 1
        else:
            seedhi_baat_count = 2
            tier2_count = 2
        eng_count = "1"
    else:  # count >= 5
        if allow_detailed:
            seedhi_baat_count = 3
            tier2_count = 1
        else:
            seedhi_baat_count = 3
            tier2_count = 2
        eng_count = "1"

    if allow_detailed:
        length_directive = (
            f"MANDATORY TIER LENGTH CONSTRAINTS ({count} reviews total):\n"
            f"- Tier 1 — SEEDHI BAAT: Exactly {seedhi_baat_count} review(s) MUST be ultra-short (4–10 words max). "
            f"Direct one-thumb reactions — e.g. 'same as shown in picture', 'cooling boht fit hy', 'mashallah achi machine hy', 'Good quality recommended'. NEVER write more than 10 words for these {seedhi_baat_count} reviews!\n"
            f"- Tier 2 — PRACTICAL SHORT: Exactly {tier2_count} review(s) (12–20 words). 1-2 run-on sentences with a practical observation or minor 4-star friction.\n"
            f"- Tier 3 — DETAILED EXPERIENCE: Exactly 1 review (25–35 words max). Short household context. NEVER an essay. ZERO 'dar tha' clichés."
        )
    else:
        length_directive = (
            f"MANDATORY TIER LENGTH CONSTRAINTS ({count} reviews total):\n"
            f"- Tier 1 — SEEDHI BAAT: Exactly {seedhi_baat_count} review(s) MUST be ultra-short (4–10 words max). "
            f"Direct one-thumb reactions — e.g. 'same as shown in picture', 'cooling boht fit hy', 'mashallah achi machine hy', 'Good quality recommended'. NEVER write more than 10 words for these {seedhi_baat_count} reviews!\n"
            f"- Tier 2 — PRACTICAL SHORT: Exactly {tier2_count} review(s) (12–20 words). 1-2 run-on sentences with a practical observation or minor 4-star friction.\n"
            f"- NO TIER 3: All reviews in this batch must stay in Tier 1 and Tier 2 only."
        )

    # --- Pre-generate dynamic unique Pakistani reviewer names for this batch ---
    assigned_names: list[str] = []
    while len(assigned_names) < count:
        cand = generate_reviewer().name
        if cand not in assigned_names:
            assigned_names.append(cand)
    assigned_names_str = ", ".join(f'"{n}"' for n in assigned_names)

    # --- Detect product category and get focused vocabulary ---
    category = detect_category(product_name)
    vocab_guidance = get_category_vocab(category)
    component_info = COMPONENT_GROUNDING.get(category, "")

    # --- Build focused product context block with strict anti-copy mandate ---
    product_context = f"""<product_context>
Product: {product_name}
Category: {category}
Physical reality: {component_info}
Vocabulary guidance (THEME INSPIRATION ONLY — DO NOT COPY PHRASES VERBATIM):
{vocab_guidance}
CRITICAL ANTI-COPY MANDATE: The vocabulary guidance above gives you topic ideas only. NEVER copy these exact phrases, sentence structures, or wording word-for-word. Express these ideas using fresh, unique sentence structures and diverse words.
IMPORTANT: Only mention parts and experiences that physically exist on THIS product. Never cross-contaminate from other product types.
</product_context>"""

    # --- Dynamic focus angle assignment for within-batch diversity ---
    angles_pool = CATEGORY_ANGLES.get(category, CATEGORY_ANGLES["generic"]).copy()
    random.shuffle(angles_pool)
    selected_angles = [angles_pool[i % len(angles_pool)] for i in range(count)]
    angles_formatted = "\n".join(
        f"- Review {i+1} ({assigned_names[i]}): Focus angle -> {selected_angles[i]}"
        for i in range(count)
    )
    diversity_directive = (
        "<diversity_steering>\n"
        "Each review in this batch MUST explore a distinct topic angle:\n"
        f"{angles_formatted}\n"
        "Do NOT write multiple reviews talking about the exact same thing.\n"
        "</diversity_steering>\n\n"
    )

    # --- Anti-repetition memory directive (Cross-product diversity) ---
    anti_rep_items: list[str] = []
    if rejected_reviews:
        for r in rejected_reviews:
            r_clean = r.strip()
            if r_clean and r_clean not in anti_rep_items:
                anti_rep_items.append(f"[DISCARDED PREVIOUS ATTEMPT]: {r_clean}")
    if recent_reviews:
        for r in recent_reviews:
            r_clean = r.strip()
            if r_clean and r_clean not in anti_rep_items:
                anti_rep_items.append(r_clean)

    if anti_rep_items:
        formatted_past = "\n".join(f'- "{item}"' for item in anti_rep_items[:12])
        anti_repetition_directive = (
            "<anti_repetition_memory>\n"
            "CRITICAL DIVERSITY MANDATE:\n"
            "The following reviews were recently published or discarded for products in our store:\n"
            f"{formatted_past}\n\n"
            "YOU MUST NOT REPEAT OR CLOSELY PARAPHRASE ANY OF THESE REVIEWS!\n"
            "- Do NOT use the same opening words, sentence structures, or punchlines.\n"
            "- Do NOT repeat the exact same complaint or observation.\n"
            "- Make your new reviews 100% distinct in voice, word choice, and angle.\n"
            "</anti_repetition_memory>\n\n"
        )
    else:
        anti_repetition_directive = ""

    # --- Brand-specific logistics directive (programmatically enforced) ---
    name_lower = product_name.lower()
    is_major_brand = any(b in name_lower for b in MAJOR_APPLIANCE_BRANDS)

    if is_major_brand:
        logistics_directive = (
            "<brand_logistics>\n"
            "MAJOR BRAND — Karachi/Lahore may get same-day delivery.\n"
            "Other cities: 2–3 day standard delivery.\n"
            "Same-day mention: at most ONE review in this batch, keep it short: 'Same day delivery mil gayi, highly recommended.'\n"
            "</brand_logistics>"
        )
    else:
        logistics_directive = (
            "<brand_logistics>\n"
            "STANDARD COURIER — No same-day delivery for this brand.\n"
            "BANNED: 'same day', 'same day delivery', 'subah order shaam ko mila'.\n"
            "Use only: '2-3 din mein deliver hua', 'parcel safe mila', 'time par delivery mil gayi'.\n"
            "</brand_logistics>"
        )

    # --- Optional free installation directive ---
    if include_installation:
        installation_directive = (
            "<free_installation>\n"
            "Include ONE brief free installation mention in a 5-STAR review only.\n"
            "e.g. 'Delivery ke sath installation bhi free kar di, boht achi service hai' or 'Free installation mil gayi, thanks kiachahye!'\n"
            "</free_installation>\n"
        )
    else:
        installation_directive = ""

    # --- Optional customer service directive (dynamically varied) ---
    if include_customer_service:
        cs_angles = [
            "support team ne guide kiya helpful thay",
            "helpline walon ne jaldi jawab diya",
            "after sale service achi mili",
            "query ka response acha aur quick tha",
            "customer care walon se baat hui madad kar di unhon ne",
            "seller chat pe response quick mil gaya tha",
        ]
        picked_cs = random.choice(cs_angles)
        cs_directive = (
            "<customer_service>\n"
            f"Include ONE brief customer service nod in a 5-STAR review only (phrasing angle: '{picked_cs}').\n"
            "NEVER write long stories about contacting support.\n"
            "</customer_service>\n"
        )
    else:
        cs_directive = ""

    return f"""<task>
<product_title>{product_name}</product_title>
<sku>{clean_sku}</sku>
<target_ratings>{ratings}</target_ratings>

{product_context}

{logistics_directive}

{installation_directive}{cs_directive}{anti_repetition_directive}{diversity_directive}<batch_requirements>
Generate exactly {count} reviews. Match each review to its corresponding rating in <target_ratings>.
Assign these exact reviewer names to the {count} reviews: {assigned_names_str}

Length distribution:
{length_directive}

Language: Write {eng_count} review(s) in Pakistani English. STUDY these REAL examples of how Pakistanis write English reviews on Daraz:
- "I am satisfied to this product. Too good and brand new product. Thanku KiaChahye"
- "good product came in secure packing. Definitely recommend this product"
- "Good quality. Reached within one day. Complete in original condition. Working perfectly."
- "Very nice product same as shown in pictures"
- "the fridge is excellent in terms of cooling and quality. received on time"
- "Overall a good product and looks fine but noise sound hay but not too much"
- "product is really fantastic but the delivery was late. overall experience is pretty good."
NOTICE: Pakistani English uses short sentences WITH periods, has typos ("recived", "Recommnded", "Thanku"), broken prepositions ("satisfied TO"), drops articles, and sometimes mixes in Urdu words ("hay", "hai"). NEVER write clean Amazon-style English.
Remaining reviews in natural Roman Urdu.
</batch_requirements>

<think_first>
Before writing each review, imagine:
1. WHO is typing this? Look at their assigned name: is this person an aurat (woman) or mard (man)? Match their voice and verbs accordingly (female: "khush hui", male: "khush hua"). If male name on female styling product: bought for sister/mother/family!
2. Are they typing on phone quickly or carefully? Do they use "hy" or "hai"? Short or long?
3. What physical parts does this product actually have? (don't mention parts that don't exist)
4. What would they ACTUALLY notice first — packaging? product look? first use?
5. For 4-star: pick a DIFFERENT minor complaint than the other 4-star reviews
6. Read your review back — does it sound like a WhatsApp voice-to-text or a product manual? If manual, rewrite it messier.
7. DIVERSITY CHECK: Compare your proposed review against <anti_repetition_memory> and other reviews in this batch. If it uses identical words or starts the same way, rewrite it with a completely different vocabulary and angle.
</think_first>

<instructions>
Write {count} reviews that sound like real mobile-typed Daraz.pk buyer feedback.
Each review must have a completely different structure, opening, and angle.
Return ONLY a raw JSON array: [{{"name": "string", "rating": number, "review": "string"}}]
</instructions>
</task>"""
