import os
from pathlib import Path

import pandas as pd

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

ENG = 'Likes/Followers/Visits/Downloads'
CHARTS_ENABLED = True  # switched off automatically during the full analysis run
BASE_DIR = Path(__file__).resolve().parent

PRETTY = {
    'facebook': 'Facebook', 'twitter': 'Twitter', 'youtube': 'YouTube',
    'tumblr': 'Tumblr', 'linkedin': 'LinkedIn', 'foursquare': 'Foursquare',
    'iphone app': 'iPhone App', 'iphone': 'iPhone App', 'wordpress': 'WordPress',
    'flickr': 'Flickr', 'newsletter': 'Newsletter', 'pinterest': 'Pinterest',
    'sms': 'SMS', 'android': 'Android', 'instagram': 'Instagram',
    'foursquare (badge unlock)': 'Foursquare', 'google+': 'Google+',
}


# ---------------------------------------------------------------- data load
def load_data():
    for p in (BASE_DIR / 'cleaned_customer.csv', BASE_DIR / 'data' / 'cleaned_customer.csv'):
        if p.exists():
            df = pd.read_csv(p)
            break
    else:
        raise FileNotFoundError(
            "cleaned_customer.csv not found. Place it next to analysis.py "
            "or inside a 'data' folder."
        )

    df['Date Sampled'] = pd.to_datetime(df['Date Sampled'], errors='coerce')
    df['Platform'] = df['Platform'].str.strip()

    # 'total' rows are summary rows, not real platforms -> remove them
    n_total = (df['Platform'].str.lower() == 'total').sum()
    df = df[df['Platform'].str.lower() != 'total'].copy()

    # standardize platform spellings (iPhone / iPhone App, foursquare variants)
    low = df['Platform'].str.lower()
    df['Platform'] = low.map(PRETTY).fillna(df['Platform'])

    if 'index' in df.columns:
        df = df.drop(columns='index')

    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns "
          f"(removed {n_total} 'total' rows, standardized platform names)")
    return df


def latest_per_account(df):
    """Likes/followers are cumulative, so use the latest value for each account."""
    d = df.dropna(subset=[ENG]).sort_values('Date Sampled')
    return d.drop_duplicates(subset=['Agency', 'Platform', 'Url'], keep='last')


# ------------------------------------------------------------------ helpers
def clear():
    os.system('cls' if os.name == 'nt' else 'clear')


def header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def ask_chart(series, title, kind='bar'):
    if not HAS_PLT or not CHARTS_ENABLED:
        return
    if input("\nDo you want to see a chart? (y/n): ").strip().lower() != 'y':
        return
    ax = series.plot(kind=kind, figsize=(10, 5), title=title)
    ax.set_xlabel('')
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------- analyses
def overview(df):
    header("DATASET OVERVIEW")
    print("Shape:", df.shape)
    print("\nFirst 10 rows:")
    print(df.head(10).to_string())
    print()
    df.info()


def stats(df):
    header("STATISTICAL SUMMARY")
    print(df.describe().to_string())


def missing(df):
    header("MISSING VALUES")
    m = df.isnull().sum()
    out = pd.DataFrame({'Missing': m, 'Percent': (m / len(df) * 100).round(2)})
    print(out.to_string())


def platform_dist(df):
    header("PLATFORM DISTRIBUTION (number of records)")
    s = df['Platform'].value_counts()
    print(s.to_string())
    ask_chart(s, 'Platform distribution')


def top_agencies_records(df):
    header("TOP 10 AGENCIES (by number of records)")
    s = df['Agency'].value_counts().head(10)
    print(s.to_string())
    ask_chart(s, 'Top 10 agencies by records')


def top_agencies_engagement(df):
    header("TOP 10 AGENCIES (total engagement, latest value per account)")
    s = latest_per_account(df).groupby('Agency')[ENG].sum().sort_values(ascending=False).head(10)
    print(s.apply(lambda x: f"{x:,.0f}").to_string())
    ask_chart(s, 'Top 10 agencies by engagement')


def platform_engagement(df):
    header("PLATFORM-WISE ENGAGEMENT (latest value per account)")
    g = latest_per_account(df).groupby('Platform')[ENG].agg(
        Accounts='count', Total='sum', Average='mean', Median='median', Max='max'
    ).sort_values('Total', ascending=False)
    print(g.round(0).to_string())
    ask_chart(g['Total'], 'Total engagement by platform')


def monthly_trend(df):
    header("MONTHLY TREND (median engagement per month)")
    d = df.dropna(subset=[ENG, 'Date Sampled'])
    s = d.groupby(d['Date Sampled'].dt.to_period('M'))[ENG].median()
    print(s.round(0).to_string())
    ask_chart(s, 'Median engagement per month', kind='line')


def top_accounts(df):
    header("TOP 10 INDIVIDUAL ACCOUNTS (by engagement)")
    t = latest_per_account(df).sort_values(ENG, ascending=False).head(10)
    print(t[['Agency', 'Platform', 'Url', ENG]].to_string(index=False))


def agency_platform_presence(df):
    header("AGENCIES USING THE MOST PLATFORMS")
    s = df.groupby('Agency')['Platform'].nunique().sort_values(ascending=False).head(10)
    print(s.to_string())
    ask_chart(s, 'Number of platforms per agency')


def search_agency(df):
    header("SEARCH AN AGENCY")
    name = input("Enter the agency name (or part of it): ").strip()
    if not name:
        print("Nothing entered.")
        return
    sub = df[df['Agency'].str.contains(name, case=False, na=False, regex=False)]
    if sub.empty:
        print("No matching agency found.")
        return
    print(f"\nMatching agencies: {sub['Agency'].unique().tolist()}")
    lat = latest_per_account(sub)
    if lat.empty:
        print("No engagement data available for this agency.")
        return
    print("\nPlatform-wise latest engagement:")
    print(lat.groupby(['Agency', 'Platform'])[ENG].sum().apply(lambda x: f"{x:,.0f}").to_string())


def date_range(df):
    header("DATE RANGE & SAMPLING")
    print("Start:", df['Date Sampled'].min())
    print("End  :", df['Date Sampled'].max())
    print("\nRecords per month:")
    print(df.groupby(df['Date Sampled'].dt.to_period('M')).size().to_string())
    print("\nNote: if a row's date looks very different from its neighbouring "
          "dates (e.g. 2012-12-12 between 2011 dates), it may be a typo.")


def run_all(df):
    """Runs every analysis (except the interactive agency search) in one go."""
    global CHARTS_ENABLED
    CHARTS_ENABLED = False  # no chart prompts in the middle of the full run
    steps = [overview, stats, missing, platform_dist, top_agencies_records,
             top_agencies_engagement, platform_engagement, monthly_trend,
             top_accounts, agency_platform_presence, date_range]
    try:
        for i, fn in enumerate(steps, 1):
            print(f"\n\n##### STEP {i} of {len(steps)} #####")
            fn(df)
    finally:
        CHARTS_ENABLED = True
    header("FULL ANALYSIS COMPLETED")
    print("Tip: use option 11 to look up a specific agency, "
          "or options 4-10 to see charts.")


# --------------------------------------------------------------------- menu
MENU = {
    '1': ("Dataset overview (shape, head, info)", overview),
    '2': ("Statistical summary", stats),
    '3': ("Missing values", missing),
    '4': ("Platform distribution", platform_dist),
    '5': ("Top 10 agencies by records", top_agencies_records),
    '6': ("Top 10 agencies by engagement", top_agencies_engagement),
    '7': ("Platform-wise engagement (total / avg / median)", platform_engagement),
    '8': ("Monthly engagement trend", monthly_trend),
    '9': ("Top 10 individual accounts", top_accounts),
    '10': ("Agencies using the most platforms", agency_platform_presence),
    '11': ("Search a specific agency", search_agency),
    '12': ("Date range & records per month", date_range),
    '13': ("RUN FULL ANALYSIS (everything at once)", run_all),
}


def show_menu():
    header("SOCIAL MEDIA ENGAGEMENT ANALYSIS")
    for k, (label, _) in MENU.items():
        print(f"  {k:>2}. {label}")
    print("   0. Exit")


def main():
    df = load_data()
    input("\nPress Enter to continue...")

    while True:
        clear()
        show_menu()
        choice = input("\nEnter your choice number: ").strip()

        if choice == '0':
            print("\nGoodbye! Analysis complete.")
            break
        if choice not in MENU:
            input("Invalid choice! Press Enter to choose again...")
            continue

        clear()
        MENU[choice][1](df)
        input("\nPress Enter to choose again...")


if __name__ == '__main__':
    main()
