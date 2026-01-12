import pandas as pd
import plotly.express as px
import kaleido
import re

# Download Chrome for Kaleido (required for image export)
kaleido.get_chrome_sync()

# LOAD LIAR DATASET
filepath = "Change this to your LIAR dataset path/liar_dataset.tsv"

liar_data = pd.read_csv(filepath, sep='\t', header=None)
liar_data.columns = [
    'id', 'label', 'text', 'subject', 'speaker', 'job_title', 'state_info', 'party',
    'barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts',
    'pants_on_fire_counts', 'location'
]

# Keep only rows with a location
liar_news = liar_data.dropna(subset=['location']).reset_index(drop=True)

# Filter fake news labels for false, pants-fire, barely-true
fake_labels = ["false", "pants-fire", "barely-true"]
fake_news = liar_news[liar_news["label"].isin(fake_labels)].reset_index(drop=True)

# MAPPING US state names to abbreviations so we can plot on choropleth US map
#chatgpt "Create a dictionary mapping US state names to their two letter abbreviations"
us_state_abbrev = {
    'Alabama':'AL','Alaska':'AK','Arizona':'AZ','Arkansas':'AR','California':'CA',
    'Colorado':'CO','Connecticut':'CT','Delaware':'DE','Florida':'FL','Georgia':'GA',
    'Hawaii':'HI','Idaho':'ID','Illinois':'IL','Indiana':'IN','Iowa':'IA',
    'Kansas':'KS','Kentucky':'KY','Louisiana':'LA','Maine':'ME','Maryland':'MD',
    'Massachusetts':'MA','Michigan':'MI','Minnesota':'MN','Mississippi':'MS','Missouri':'MO',
    'Montana':'MT','Nebraska':'NE','Nevada':'NV','New Hampshire':'NH','New Jersey':'NJ',
    'New Mexico':'NM','New York':'NY','North Carolina':'NC','North Dakota':'ND','Ohio':'OH',
    'Oklahoma':'OK','Oregon':'OR','Pennsylvania':'PA','Rhode Island':'RI','South Carolina':'SC',
    'South Dakota':'SD','Tennessee':'TN','Texas':'TX','Utah':'UT','Vermont':'VT','Virginia':'VA',
    'Washington':'WA','West Virginia':'WV','Wisconsin':'WI','Wyoming':'WY','District Of Columbia':'DC'
}

# Extract state codes from location strings
def extract_state(loc):
    if pd.isna(loc):
        return None
    loc = loc.strip()
    
    # Check for two-letter state code at end (e.g., "Seattle, WA")
    match = re.search(r'([A-Z]{2})$', loc)
    if match and match.group(1) in us_state_abbrev.values():
        return match.group(1)
    
    # Check for full state names first
    for state, code in us_state_abbrev.items():
        if state.lower() in loc.lower():
            return code
    
    # Handle edge cases
    if "Washington D.C." in loc or "District of Columbia" in loc:
        return "DC"
    if loc == "Washington":  # assume state
        return "WA"
    
    return None

# Apply extraction
fake_news.loc[:, 'state_code'] = fake_news['location'].apply(extract_state)

# Drop unmapped locations
fake_news = fake_news[fake_news['state_code'].notna()]

# Group by state and count fake news instances
state_counts = fake_news.groupby('state_code').size().reset_index(name='fake_count')


# Plot a US map with fake news counts
fig = px.choropleth(
    state_counts,
    locations='state_code',      
    locationmode='USA-states',   
    color='fake_count',          
    color_continuous_scale='Reds',
    scope='usa',
    title='Geographical Distribution of Fake News (LIAR Dataset)'
)


# Save as a PNG image
fig.write_image("fake_news_us_map.png", width=1200, height=800)

print("Map saved as fake_news_us_map.png")
