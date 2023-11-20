import matplotlib.pyplot as plt

def plot_modalities(dataframe, subject, modalities, segment=0, num_rows=3, num_cols=3, figsize=(10, 10), common_title=None):
    """
    Plot modalities of a subject's data in subplots.

    Parameters:
        - dataframe: The dataframe containing the data.
        - subject: The specified subject.
        - modalities: The list of modalities to plot.
        - segment: The specified segment of the data (default: 0).
        - num_rows: The number of rows in the subplot grid (default: 3).
        - num_cols: The number of columns in the subplot grid (default: 3).
        - figsize: The size of the figure (default: (10, 10)).
        - common_title: The common title to add above the subplots (default: None).

    Returns:
        None
    """
    # Filter dataframe for the specified subject
    df_subject = dataframe[dataframe['Subject'] == subject]

    # Create subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    # Flatten axes to iterate over them
    axes = axes.flatten()

    # Plot each modality
    for i, modality in enumerate(modalities):
        # Extract the data for the specified segment
        data_60_sec = df_subject['Data'].iloc[segment][:-3][:, i]

        # Plot the data with improved graphical elements
        axes[i].plot(data_60_sec, color='red', linestyle='-', marker='o', markersize=3, label=modality)
        axes[i].set_title(f'{modality} - {subject}')
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('Value')
        axes[i].grid(True)
        axes[i].legend()
        
    # Add a common title above the subplots
    if common_title:
        plt.suptitle(common_title, fontsize=16)   

    # Remove extra subplots if there are fewer modalities than expected
    for j in range(len(modalities), num_rows * num_cols):
        fig.delaxes(axes[j])

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()
