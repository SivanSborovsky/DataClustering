# categorizing words manually
#TODO: add to this database
god = {'יהוה','האלוהים','ואלוהים','אלוהים'}
creation = { 'ברא','יהי','ויהי-אור','ויהי-ערב','ויהי','בהיבראם','ויהי-בוקר'}
skies = {'השמיים','רקיע','ורוח','את-הרקיע','לרקיע','והירח','כוכבים','השמש','ושמיים'}
land = { 'והארץ','האדמה','וארץ','הארץ','ארצה','בארץ'}
chaos = { 'תוהו', 'ובוהו','חמס','המבול','תהום'}
darkness = {'וחושך','החושך','לילה','ולחושך','והירח'}
light = {'אור','את-האור','האור','לאור', 'יום'}
water = {'המים','מים', 'בנחל','למים','בארות'}
nature = {'המים','השדה','שיח','יצמח','עשב'}
conversation = {'ויאמר','לאמור','ויספר'}
good = {'כי-טוב','טוב'}
family = {'אבי','בני','בן','לאחיו','אל-אביו','מאביהן','אביו','ואימך','ואל-אחיו','ואחיך','אחיו','אשתך','את-אשתך','ואביו'}
community = {'העם','עמי'}
name = {'שמו','ותקרא','שמות','כשמות'}
people = {'העם','מואב','והכנעני','פלשתים'}
birth = {'ותהרינה','ותלד'}
dreams = {'חלום','חלמת','החלום','חלמתי'}
death = {'מות','יומת'}
blessings = {'ואברכך','ברכה','ויברכהו','מברכיך'}
life = {'חיים','נשמת','להחיות'}
animals = {'הבהמה','כאריה','חית','וחמורים','צאן','מהעוף','צאן-ובקר','ואתונות','וגמלים','ובקר'}

dictionaries = [god, creation,skies,land,chaos,darkness,light,water,nature,conversation,good,family,community,name,people,birth,dreams,death,blessings,life,animals]
dict_names = ["god", "creation","skies","land","chaos","darkness","light","water","nature","conversation","good","family","community","name","people","birth","dreams","death","blessings","life","animals"]

def get_dicts():return dictionaries
def get_dicts_names(): return dict_names