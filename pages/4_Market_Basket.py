import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from data_loader import load_data
import os

st.set_page_config(layout="wide")
st.title("Module V: Frequent Pattern Mining")

st.header("Association Rule Theory")
st.markdown("""
Market Basket Analysis is a modeling technique based upon the theory that if you buy a certain 
group of items, you are more or less likely to buy another group of items. In Business 
Intelligence, this is used to optimize store layouts, cross-selling strategies, and 
promotional bundles.
""")

st.latex(r"Support(A \cup B) = P(A \cap B)")
st.latex(r"Confidence(A \rightarrow B) = \frac{P(A \cap B)}{P(A)}")
st.latex(r"Lift(A \rightarrow B) = \frac{Confidence(A \rightarrow B)}{Support(B)}")

df = load_data()

basket = df.groupby(['Order ID', 'Sub-Category'])['Quantity'].sum().unstack().reset_index().fillna(0).set_index('Order ID')

basket_bool = (basket > 0)

st.header("Mining Configuration")
st.markdown("Adjust thresholds to refine the strength and frequency of identified associations.")

col_a, col_b = st.columns(2)
with col_a:
    support_val = st.slider("Minimum Support Threshold", 0.01, 0.029, step=0.005, format="%.3f")
with col_b:
   lift_val = st.slider("Minimum Lift (Association Strength)", 0.1, 1.0, 0.1, step=0.1)

frequent_itemsets = apriori(basket_bool, min_support=support_val, use_colnames=True)

if not frequent_itemsets.empty:
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=lift_val)
    
    if not rules.empty:
        display_rules = rules.copy()
        
        display_rules = display_rules.sort_values('lift', ascending=False).head(20)

        display_rules['Product_A'] = display_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        display_rules['Product_B'] = display_rules['consequents'].apply(lambda x: ', '.join(list(x)))

        st.header("Mining Results and Key Insights")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Frequent Itemsets", len(frequent_itemsets))
        m2.metric("Significant Rules", len(rules))
        m3.metric("Max Lift Found", f"{rules['lift'].max():.2f}")

        st.markdown("#### Identified Association Rules")
        st.dataframe(
            display_rules[['Product_A', 'Product_B', 'support', 'confidence', 'lift']],
            use_container_width=True,
            column_config={
                "Product_A": "If Customer Buys...",
                "Product_B": "...They are likely to buy",
                "support": st.column_config.NumberColumn("Support", format="%.4f"),
                "confidence": st.column_config.NumberColumn("Confidence", format="%.4f"),
                "lift": st.column_config.NumberColumn("Lift Strength", format="%.2f")
            }
        )

        st.divider()
        st.header("Business Intelligence Decisions")
        st.markdown("""
        Based on the high-lift rules identified above, the following BI decisions are recommended:
        1. **Product Bundling**: Items in 'Product A' and 'Product B' should be sold as a discounted package.
        2. **Store Layout**: Physical or digital proximity of these items should be increased to facilitate cross-buying.
        3. **Inventory Management**: Stocks of 'Product B' should be correlated with promotional campaigns for 'Product A'.
        """)

        if st.button("Archive Mining Results"):
            with open('bi_project_results.txt', 'a') as f:
                f.write(f"\nMODULE V ASSOCIATION MINING (Support: {support_val}, Lift: {lift_val})\n")
                f.write(display_rules[['Product_A', 'Product_B', 'lift']].to_string() + "\n")
            st.success("Analysis archived to bi_project_results.txt")
    else:
        st.error(f"No association rules found with a Lift threshold of {lift_val}. Lower the Lift slider to 1.0.")
else:
    st.error(f"No frequent itemsets found at Support {support_val}. Lower the Support slider.")