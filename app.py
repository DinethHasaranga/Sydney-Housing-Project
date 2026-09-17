import streamlit as st
import pandas as pd
import joblib
from PIL import Image


# Page configuration
st.set_page_config(
    page_title="Sydney Housing Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# Load the trained model
model = joblib.load("sydney_property_price_model.pkl")

# Hero image
# Load the hero image
hero_image = Image.open("hero.jpg")

# Crop the image to a wide banner
width, height = hero_image.size

target_ratio = 3.5  # wide hero banner
current_ratio = width / height

if current_ratio > target_ratio:
    # Image is too wide, crop left and right
    new_width = int(height * target_ratio)
    left = (width - new_width) // 2
    hero_image = hero_image.crop(
        (left, 0, left + new_width, height)
    )

else:
    # Image is too tall, crop top and bottom
    new_height = int(width / target_ratio)
    top = (height - new_height) // 2
    hero_image = hero_image.crop(
        (0, top, width, top + new_height)
    )

# Resize to a consistent hero size
hero_image = hero_image.resize(
    (1600, 450)
)

# Display the hero image
st.image(
    hero_image,
    use_container_width=True
)

# Title and introduction
st.title("Sydney Housing Price Prediction")

st.write(
    "Enter the property characteristics below to estimate "
    "the expected sale price using the tuned Random Forest model."
)

st.divider()

# Main input section
left_col, right_col = st.columns(2)

# Left column - Property details
with left_col:

    st.subheader("Property Details")

    suburb = st.selectbox(
        "Suburb",
        [
            "Burwood",
            "Parramatta",
            "Campbelltown"
        ]
    )

    property_type = st.selectbox(
        "Property Type",
        [
            "House",
            "Apartment",
            "Unit",
            "Townhouse",
            "Villa",
            "Duplex/semi-detached"
        ]
    )

    bedrooms = st.number_input(
        "Bedrooms",
        min_value=1,
        max_value=10,
        value=3,
        step=1
    )

    bathrooms = st.number_input(
        "Bathrooms",
        min_value=1,
        max_value=10,
        value=2,
        step=1
    )

    car_spaces = st.number_input(
        "Car Spaces",
        min_value=0,
        max_value=10,
        value=1,
        step=1
    )

# Right column - Size and sale information
with right_col:

    st.subheader("Sale and Property Information")

    property_size_sqm = st.number_input(
        "Property Size (sqm)",
        min_value=1.0,
        value=500.0,
        step=10.0
    )

    sale_year = st.selectbox(
        "Sale Year",
        [
            2024,
            2025,
            2026
        ],
        index=2
    )

    sale_quarter = st.selectbox(
        "Sale Quarter",
        [
            1,
            2,
            3,
            4
        ]
    )


# Additional property features
st.divider()

st.subheader("Additional Property Features")

feature_col1, feature_col2, feature_col3, feature_col4 = st.columns(4)

with feature_col1:
    has_pool = st.checkbox(
        "Swimming Pool"
    )

with feature_col2:
    renovated = st.checkbox(
        "Renovated"
    )

with feature_col3:
    has_solar = st.checkbox(
        "Solar Features"
    )

with feature_col4:
    development_potential = st.checkbox(
        "Development Potential"
    )


# Prediction section
st.divider()

predict_col1, predict_col2, predict_col3 = st.columns(
    [1, 2, 1]
)

with predict_col2:

    if st.button(
        "Predict Sale Price",
        use_container_width=True
    ):

        # Create input dataframe with the same
        # predictors used during model training

        input_data = pd.DataFrame({
            "suburb": [suburb],
            "property_type": [property_type],
            "bedrooms": [bedrooms],
            "bathrooms": [bathrooms],
            "car_spaces": [car_spaces],
            "property_size_sqm": [property_size_sqm],
            "sale_year": [sale_year],
            "sale_quarter": [sale_quarter],
            "has_pool": [int(has_pool)],
            "renovated": [int(renovated)],
            "has_solar": [int(has_solar)],
            "development_potential": [
                int(development_potential)
            ]
        })

        # Generate prediction
        prediction = model.predict(
            input_data
        )[0]

        # Display prediction
        st.success(
            "Prediction generated successfully."
        )

        st.metric(
            label="Estimated Sale Price",
            value=f"{prediction:,.0f} dollars"
        )


