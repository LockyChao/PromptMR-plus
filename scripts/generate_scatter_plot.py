import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def determine_mapping_modality(file_path, original_modality):
    """
    Determine the specific mapping modality based on file path content
    
    Args:
        file_path (str): The file path from the File column
        original_modality (str): The original modality value
    
    Returns:
        str: The specific modality (T1map, T2map, T1mapposy, or original modality)
    """
    if pd.isna(file_path) or original_modality != 'Mapping':
        return original_modality
    
    file_path_lower = str(file_path).lower()
    
    # Check for T1mapposy first (more specific)
    if 't1mappost' == file_path_lower:
        return 'T1mappost'
    # Check for T1map
    elif 't1map' == file_path_lower:
        return 'T1map'
    # Check for T2map
    elif 't2map' == file_path_lower:
        return 'T2map'
    else:
        # If it's Mapping but doesn't match any specific type, keep as Mapping
        return original_modality

def read_excel_and_plot(excel_file_path, output_path='scatter_plot.png'):
    """
    Read Excel file and create scatter plot with SSIM vs 90th_Percentile grouped by Modality
    
    Args:
        excel_file_path (str): Path to the Excel file
        output_path (str): Path to save the PNG plot
    """
    try:
        # Read Excel file
        print(f"Reading Excel file: {excel_file_path}")
        df = pd.read_excel(excel_file_path)
        
        # Check if required columns exist
        required_columns = ["Modality", "SSIM", "90th_Percentile", "File"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Error: Missing columns in Excel file: {missing_columns}")
            print(f"Available columns: {list(df.columns)}")
            return
        
        # Extract the required columns
        plot_data = df[required_columns].copy()
        
        # Create new modality column with separated mapping types
        plot_data['Detailed_Modality'] = plot_data.apply(
            lambda row: determine_mapping_modality(row['File'], row['Modality']), 
            axis=1
        )
        
        # Remove rows with missing values in the key columns for plotting
        plot_data = plot_data.dropna(subset=['Detailed_Modality', 'SSIM', '90th_Percentile'])
        
        if plot_data.empty:
            print("Error: No valid data found after removing missing values")
            return
        
        print(f"Data shape: {plot_data.shape}")
        print(f"Unique detailed modalities: {plot_data['Detailed_Modality'].unique()}")
        
        # Split data based on 90th_Percentile threshold
        data_above_1 = plot_data[plot_data['90th_Percentile'] > 1]
        data_below_1 = plot_data[plot_data['90th_Percentile'] < 1]
        
        print(f"Data points with 90th_Percentile > 1: {len(data_above_1)}")
        print(f"Data points with 90th_Percentile < 1: {len(data_below_1)}")
        
        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Get unique modalities and assign colors (consistent across both plots)
        unique_modalities = plot_data['Detailed_Modality'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_modalities)))
        color_map = {modality: colors[i] for i, modality in enumerate(unique_modalities)}
        
        # Plot 1: 90th_Percentile > 1
        for modality in unique_modalities:
            modality_data = data_above_1[data_above_1['Detailed_Modality'] == modality]
            if not modality_data.empty:
                ax1.scatter(
                    modality_data['SSIM'],
                    modality_data['90th_Percentile'],
                    c=[color_map[modality]],
                    label=modality,
                    s=60,
                    alpha=0.7
                )
        
        ax1.set_title('90th Percentile > 1', fontsize=14, fontweight='bold')
        ax1.set_xlabel('SSIM', fontsize=12)
        ax1.set_ylabel('90th Percentile', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(title='Detailed Modality', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 2: 90th_Percentile < 1
        for modality in unique_modalities:
            modality_data = data_below_1[data_below_1['Detailed_Modality'] == modality]
            if not modality_data.empty:
                ax2.scatter(
                    modality_data['SSIM'],
                    modality_data['90th_Percentile'],
                    c=[color_map[modality]],
                    label=modality,
                    s=60,
                    alpha=0.7
                )
        
        ax2.set_title('90th Percentile < 1', fontsize=14, fontweight='bold')
        ax2.set_xlabel('SSIM', fontsize=12)
        ax2.set_ylabel('90th Percentile', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(title='Detailed Modality', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add overall title
        fig.suptitle('Scatter Plot: SSIM vs 90th Percentile by Detailed Modality', fontsize=16, fontweight='bold')
        
        # Adjust layout to prevent legend cutoff
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved as: {output_path}")
        
        # Display basic statistics
        print("\nData Summary:")
        print("\n--- All Data ---")
        print(plot_data.groupby('Detailed_Modality').agg({
            'SSIM': ['count', 'mean', 'std'],
            '90th_Percentile': ['mean', 'std']
        }).round(4))
        
        if not data_above_1.empty:
            print("\n--- 90th Percentile > 1 ---")
            print(data_above_1.groupby('Detailed_Modality').agg({
                'SSIM': ['count', 'mean', 'std'],
                '90th_Percentile': ['mean', 'std']
            }).round(4))
        
        if not data_below_1.empty:
            print("\n--- 90th Percentile < 1 ---")
            print(data_below_1.groupby('Detailed_Modality').agg({
                'SSIM': ['count', 'mean', 'std'],
                '90th_Percentile': ['mean', 'std']
            }).round(4))
        
        plt.close()
        
    except FileNotFoundError:
        print(f"Error: Excel file not found at {excel_file_path}")
    except Exception as e:
        print(f"Error: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Generate scatter plot from Excel data')
    parser.add_argument('excel_file', help='Path to the Excel file')
    parser.add_argument('--output', '-o', default='scatter_plot.png', 
                       help='Output PNG file path (default: scatter_plot.png)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.excel_file):
        print(f"Error: Excel file does not exist: {args.excel_file}")
        return
    
    read_excel_and_plot(args.excel_file, args.output)

if __name__ == "__main__":
    main()