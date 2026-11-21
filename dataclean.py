
import numpy as np
import pandas as pd

# Load the messy CSV dataset
df = pd.read_csv('SocialMediaUsageDataset.csv')

print("Original Dataset:")
print(df.head(10))

print("\nDataset Information:")
print(df.info())

# 1. Remove duplicate records
df = df.drop_duplicates()

# 2. Clean Agency column
df['Agency'] = (
    df['Agency']
    .astype('string')
    .str.strip()
)

# 3. Clean and standardize Platform column
df['Platform'] = (
    df['Platform']
    .astype('string')
    .str.strip()
    .str.lower()
)

platform_map = {
    'youtube': 'YouTube',
    'facebook': 'Facebook',
    'twitter': 'Twitter',
    'iphone app': 'iPhone App',
    'iphone': 'iPhone',
    'wordpress': 'WordPress',
    'linkedin': 'LinkedIn',
    'linked-in': 'LinkedIn',
    'tumblr': 'Tumblr',
    'instagram': 'Instagram',
    'foursquare': 'Foursquare'
}

df['Platform'] = df['Platform'].replace(platform_map)

# 4. Clean URL column
df['Url'] = (
    df['Url']
    .astype('string')
    .str.strip()
)

df['Url'] = df['Url'].replace('', pd.NA)

# 5. Clean Date Sampled column
df['Date Sampled'] = pd.to_datetime(
    df['Date Sampled'],
    format='mixed',
    errors='coerce'
)

# 6. Clean Likes/Followers/Visits/Downloads
metric = 'Likes/Followers/Visits/Downloads'

df[metric] = pd.to_numeric(
    df[metric],
    errors='coerce'
)

# Handle negative values
df.loc[df[metric] < 0, metric] = np.nan

# 7. Handle missing values
# Keep missing social media metrics as NaN
# because different platforms have different metrics.

# 8. Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 9. Check duplicate records
print("\nDuplicate Records:")
print(df.duplicated().sum())

# 10. Export cleaned dataset
df.to_csv('cleaned_customer.csv', index=False)

print("\nCleaned Dataset:")
print(df.head(10))

print("\nData Cleaning Completed Successfully!")
