#######################################################################################################################################################
# 
# Name: Pavan Kumar Anil Kumar
# SID: 750026292
# Exam Date: 
# Module:
# Github link for this assignment: https://github.com/UniversityExeterBusinessSchool/mock-test-2-Pavan8765.git
#
#######################################################################################################################################################
# Instruction 1. Read each question carefully and complete the scripts as instructed.

# Instruction 2. Only ethical and minimal use of AI is allowed. You may use AI to get advice on tool usage or language syntax, 
#                but not to generate code. Clearly indicate how and where you used AI.

# Instruction 3. Include comments explaining the logic of your code and the output as a comment below the code.

# Instruction 4. Commit to Git and upload to ELE once you finish.

#######################################################################################################################################################

# Question 1 - Loops and Lists
# You are given a list of numbers representing weekly sales in units.
# weekly_sales = [120, 85, 100, 90, 110, 95, 130]

# Write a for loop that iterates through the list and prints whether each week's sales were above or below the average sales for the period.
# Calculate and print the average sales.
weekly_sales = [120, 85, 100, 90, 110, 95, 130]

# Calculate the average sales
average_sales = sum(weekly_sales) / len(weekly_sales)
print(f"Average sales: {average_sales:.2f}")

# Iterate through the list and check if each week's sales are above or below the average
for week, sales in enumerate(weekly_sales, start=1):
    if sales > average_sales:
        print(f"Week {week}: {sales} units (Above average)")
    elif sales < average_sales:
        print(f"Week {week}: {sales} units (Below average)")
    else:
        print(f"Week {week}: {sales} units (Average)")


#######################################################################################################################################################

# Question 2 - String Manipulation
# A customer feedback string is provided:
# customer_feedback = """The product was good but could be improved. I especially appreciated the customer support and fast response times."""

# # Find the first and last occurrence of the words 'good' and 'improved' in the feedback using string methods.
# # Store each position in a list as a tuple (start, end) for both words and print the list.

customer_feedback = """The product was good but could be improved. I especially appreciated the customer support and fast response times."""

# Find the first and last occurrence of "good"
good_start = customer_feedback.find("good")
good_end = good_start + len("good") if good_start != -1 else -1

# Find the first and last occurrence of "improved"
improved_start = customer_feedback.find("improved")
improved_end = improved_start + len("improved") if improved_start != -1 else -1

# Store the positions in a list as tuples
positions = [("good", (good_start, good_end)), ("improved", (improved_start, improved_end))]

# Print the list of tuples
print(positions)


#######################################################################################################################################################

# Question 3 - Functions for Business Metrics
# Define functions to calculate the following metrics, and call each function with sample values (use your student ID digits for customization).

# 1. Net Profit Margin: Calculate as (Net Profit / Revenue) * 100.
# 2. Customer Acquisition Cost (CAC): Calculate as (Total Marketing Cost / New Customers Acquired).
# 3. Net Promoter Score (NPS): Calculate as (Promoters - Detractors) / Total Respondents * 100.
# 4. Return on Investment (ROI): Calculate as (Net Gain from Investment / Investment Cost) * 100.
# Function to calculate Net Profit Margin
def net_profit_margin(net_profit, revenue):
    margin = (net_profit / revenue) * 100
    return f"Net Profit Margin: {margin:.2f}%"

# Function to calculate Customer Acquisition Cost (CAC)
def customer_acquisition_cost(total_marketing_cost, new_customers):
    cac = total_marketing_cost / new_customers
    return f"Customer Acquisition Cost (CAC): ${cac:.2f}"

# Function to calculate Net Promoter Score (NPS)
def net_promoter_score(promoters, detractors, total_respondents):
    nps = ((promoters - detractors) / total_respondents) * 100
    return f"Net Promoter Score (NPS): {nps:.2f}%"

# Function to calculate Return on Investment (ROI)
def return_on_investment(net_gain, investment_cost):
    roi = (net_gain / investment_cost) * 100
    return f"Return on Investment (ROI): {roi:.2f}%"

# Sample values using student ID digits (let's assume your ID is 210589)
print(net_profit_margin(2105, 8900))       # Net Profit: 2105, Revenue: 8900
print(customer_acquisition_cost(5000, 210)) # Total Marketing Cost: 5000, New Customers: 210
print(net_promoter_score(58, 21, 89))       # Promoters: 58, Detractors: 21, Total Respondents: 89
print(return_on_investment(1058, 2105))     # Net Gain: 1058, Investment Cost: 2105


#######################################################################################################################################################

# Question 4 - Data Analysis with Pandas
# Using a dictionary sales_data, create a DataFrame from this dictionary, and display the DataFrame.
# Write code to calculate and print the cumulative monthly sales up to each month.


import pandas as pd

# Sales data dictionary
sales_data = {'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May'], 'Sales': [200, 220, 210, 240, 250]}

# Create a DataFrame from the dictionary
df = pd.DataFrame(sales_data)

# Display the DataFrame
print("Sales Data:")
print(df)

# Calculate cumulative monthly sales
df['Cumulative Sales'] = df['Sales'].cumsum()

# Display the updated DataFrame with cumulative sales
print("\nCumulative Monthly Sales:")
print(df)


#######################################################################################################################################################

# Question 5 - Linear Regression for Forecasting
# Using the dataset below, create a linear regression model to predict the demand for given prices.
# Predict the demand if the company sets the price at £26. Show a scatter plot of the data points and plot the regression line.

# Price (£): 15, 18, 20, 22, 25, 27, 30
# Demand (Units): 200, 180, 170, 160, 150, 140, 130

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dataset
prices = np.array([15, 18, 20, 22, 25, 27, 30]).reshape(-1, 1)
demand = np.array([200, 180, 170, 160, 150, 140, 130])

# Create a linear regression model
model = LinearRegression()
model.fit(prices, demand)

# Predict the demand if the price is £26
predicted_demand = model.predict(np.array([[26]]))

# Plotting the scatter plot and regression line
plt.figure(figsize=(8, 5))
plt.scatter(prices, demand, color='blue', label='Data points')
plt.plot(prices, model.predict(prices), color='red', label='Regression line')
plt.scatter(26, predicted_demand, color='green', marker='x', s=100, label=f'Predicted Demand at £26: {predicted_demand[0]:.2f} units')
plt.title('Price vs Demand - Linear Regression')
plt.xlabel('Price (£)')
plt.ylabel('Demand (Units)')
plt.legend()
plt.grid(True)
plt.show()

# Print predicted demand at £26
print(f"Predicted demand at £26: {predicted_demand[0]:.2f} units")


#######################################################################################################################################################

# Question 6 - Error Handling
# You are given a dictionary of prices for different products.
prices = {'A': 50, 'B': 75, 'C': 'unknown', 'D': 30}

# Write a function to calculate the total price of all items, handling any non-numeric values by skipping them.
# Include error handling in your function and explain where and why it’s needed.

def calculate_total_price(prices):
    total_price = 0  # Initialize total price
    for product, price in prices.items():
        try:
            # Try to add the price to the total (ensure it's numeric)
            total_price += float(price)
        except ValueError:
            # If a ValueError occurs (non-numeric value), skip the product
            print(f"Warning: Price for product {product} is not numeric and will be skipped.")
    return total_price

# Given dictionary of prices
prices = {'A': 50, 'B': 75, 'C': 'unknown', 'D': 30}

# Calculate and print the total price
total = calculate_total_price(prices)
print(f"Total Price: {total}")


#######################################################################################################################################################

# Question 7 - Plotting and Visualization
# Generate 50 random numbers between 1 and 500, then:
# Plot a histogram to visualize the distribution of these numbers.
# Add appropriate labels for the x-axis and y-axis, and include a title for the histogram.

import matplotlib.pyplot as plt
import random

import matplotlib.pyplot as plt
import random

# Generate 50 random numbers between 1 and 500
random_numbers = [random.randint(1, 500) for _ in range(50)]

# Plot the histogram
plt.hist(random_numbers, bins=10, edgecolor='black')

# Add labels and title
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of 50 Random Numbers Between 1 and 500')

# Display the plot
plt.show()


#######################################################################################################################################################

# # Question 8 - List Comprehensions
# # Given a list of integers representing order quantities.
# quantities = [5, 12, 9, 15, 7, 10]

# # Use a list comprehension to create a new list that doubles each quantity that is 10 or more.
# # Print the original and the new lists.
# # Given list of order quantities

quantities = [5, 12, 9, 15, 7, 10]

# List comprehension to double quantities that are 10 or more
doubled_quantities = [quantity * 2 if quantity >= 10 else quantity for quantity in quantities]

# Print the original and the new lists
print("Original List:", quantities)
print("New List:", doubled_quantities)


#######################################################################################################################################################

# Question 9 - Dictionary Manipulation
# Using the dictionary below, filter out the products with a rating of less than 4 and create a new dictionary with the remaining products.
ratings = {'product_A': 4, 'product_B': 5, 'product_C': 3, 'product_D': 2, 'product_E': 5}
# Given dictionary of product ratings
ratings = {'product_A': 4, 'product_B': 5, 'product_C': 3, 'product_D': 2, 'product_E': 5}

# Filter out products with a rating less than 4 using a dictionary comprehension
filtered_ratings = {product: rating for product, rating in ratings.items() if rating >= 4}

# Print the new dictionary
print("Filtered Dictionary:", filtered_ratings)

#######################################################################################################################################################

# Question 10 - Debugging and Correcting Code
# The following code intends to calculate the average of a list of numbers, but it contains errors:
values = [10, 20, 30, 40, 50]
total = 0
for i in values:
    total = total + i
average = total / len(values)
print("The average is" + str(average))

# Identify and correct the errors in the code.
# Comment on each error and explain your fixes.

#######################################################################################################################################################
