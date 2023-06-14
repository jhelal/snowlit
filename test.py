import pandas as pd
import statsmodels.api as sm
import numpy as np

# # Load the data from Excel
# df = pd.read_excel(r'C:\Users\james\Desktop\_PhD_Analysed_Models_Catalogue.xlsx')
#
# # Calculate additional variables
# df['AR'] = df[['Width_(m)', 'Depth_(m)']].max(axis=1) / df[['Width_(m)', 'Depth_(m)']].min(axis=1)
# df['SR'] = df['Height_(m)'] / df[['Width_(m)', 'Depth_(m)']].min(axis=1)
# df['CSW'] = df['Width_(m)'] / df['Column_Span_in_X_Dir(m)']
# df['CSD'] = df['Depth_(m)'] / df['Column_Span_in_Y_Dir(m)']
#
# # Create an Excel writer object
# with pd.ExcelWriter(r'C:\Users\james\Desktop\regression_results.xlsx') as writer:
#     # Loop over unique combinations of LLRS and LLRS_Material
#     for name, group in df.groupby(['LLRS', 'LLRS_Material']):
#         # Select relevant columns and add 'Storeys_sq' to dataframe
#         X = group[['Storeys', 'Width_(m)', 'CSW', 'Depth_(m)', 'CSD', 'AR', 'SR']].copy()
#         X['Storeys_sq'] = X['Storeys'] ** 2
#         y = group['Embodied_GHG_(kgCO2e/m^2)']
#
#         # Add constant to predictors and fit model
#         X = sm.add_constant(X)
#         model = sm.OLS(y, X)
#         results = model.fit()
#
#         # Save results to Excel
#         results_df = pd.DataFrame({
#             'coef': results.params,
#             'std err': results.bse,
#             't': results.tvalues,
#             'P>|t|': results.pvalues,
#             '[0.025': results.conf_int()[0],
#             '0.975]': results.conf_int()[1]
#         })
#
#         sheet_name = f'{name[0]}{name[1]}'
#         sheet_name = (sheet_name[:28] + '...') if len(sheet_name) > 31 else sheet_name
#         results_df.to_excel(writer, sheet_name=sheet_name)


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import statsmodels.api as sm
#
# # User specified parameters
# LLRS = "ShearWall"
# LLRS_Material = "32 MPa RC"
# Width = 35
# Column_Spans_along_Width = 5
# Depth = 35
# Column_Spans_along_Depth = 5
#
# # Load regression results
# with pd.ExcelFile(r'C:\Users\james\Desktop\regression_results.xlsx') as reader:
#     coef = pd.read_excel(reader, sheet_name=f'{LLRS}{LLRS_Material}', index_col=0)
#
# # Get coefficients
# intercept = coef.loc['const', 'coef']
# Storeys_coef = coef.loc['Storeys', 'coef']
# Storeys_sq_coef = coef.loc['Storeys_sq', 'coef']
#
# # Create storeys array
# storeys = np.arange(5, 96, 1)
#
# # Calculate predicted values
# predicted_values = intercept + Storeys_coef * storeys + Storeys_sq_coef * storeys**2
#
# # Calculate standard error of the prediction assuming a constant relative standard error
# relative_standard_error = coef['std err'].abs() / coef['coef'].abs()
# predicted_SE = predicted_values * relative_standard_error.mean()
#
# # Create figure and axes
# fig, ax = plt.subplots()
#
# # Plot predicted values
# ax.plot(storeys, predicted_values, label='Predicted values')
#
# # Plot standard errors
# ax.fill_between(storeys, predicted_values - predicted_SE, predicted_values + predicted_SE, color='gray', alpha=0.5, label='Approx. standard error')
#
# # Set labels and title
# ax.set_xlabel('Number of storeys')
# ax.set_ylabel('Embodied GHG emissions per net floor area (kgCO2e/m^2)')
# ax.set_title(f'Embodied carbon per net floor area for {LLRS} {LLRS_Material}')
#
# # Add legend
# ax.legend()
#
# # Show plot
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Constants for LLRS = "Shear Wall", LLRS_Material = "32 MPa RC"
beta0 = 85.72
beta1 = 0.06
beta2 = -2.51
beta3 = 1.66
beta4 = 1.08
beta5 = -6.76
beta6 = -6.59
beta7 = 6.84
beta8 = 22.95
R2 = 0.87

# Variables
Width = 45
Column_Spans_along_Width = 5
Depth = 45
Column_Spans_along_Depth = 5
Aspect_Ratio = max(Width, Depth) / min(Width, Depth)
Slenderness_Ratio = 1 / min(Width, Depth)

# Array of storeys
storeys = np.array(range(1, 101))

# EGHGEPNFA calculation
EGHGEPNFA = beta0 + beta1 * storeys ** 2 + beta2 * storeys + beta3 * Width + beta4 * Column_Spans_along_Width + beta5 * Depth + beta6 * Column_Spans_along_Depth + beta7 * Aspect_Ratio + beta8 * Slenderness_Ratio

# Uncertainty calculation
uncertainty = (1 - R2)
upper_bound = EGHGEPNFA * (1 + uncertainty)
lower_bound = EGHGEPNFA * (1 - uncertainty)

# Create plot
plt.figure(figsize=(10,6))
plt.plot(storeys, EGHGEPNFA, color='blue', label='Predicted EGHGEPNFA')
plt.fill_between(storeys, lower_bound, upper_bound, color='gray', alpha=0.5, label='Uncertainty')
plt.title('Predicted EGHGEPNFA with Uncertainty for Shear Wall, 32 MPa RC')
plt.xlabel('Storeys')
plt.ylabel('EGHGEPNFA')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()