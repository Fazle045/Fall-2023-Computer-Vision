from canny_edge_detection import *
'''
This is the main function that import canny_edge_detection functions, read images from currect folder based on some predefined extensions and apply canny edge detection for each
image for various sigma values

'''
def main():

    import glob

    # List of image file extensions
    image_extensions = ['*.jpg', '*.jpeg', '*.png']

    # Initialize an empty list to store the file paths
    image_list = []

    # Iterate through each extension and append matching file paths to the list
    for extension in image_extensions:
        image_list.extend(glob.glob(extension))

    # Set a list of sigma values for the experiment
    sigma_list = [0.05, 1, 10]


    for i in range(len(sigma_list)):

        # Plot the intermediate results of canny edge detector
        for filename in image_list:
            # Read image
            I = cv.imread(filename, 0)
            canny = Canny_Edge_Detection(image=I,
                                        sigma=sigma_list[i],
                                        kernel_size=5,
                                        high_th_ratio_for_canny=0.15,
                                        low_th_ratio_for_canny=0.05)
            # Generate edges of the image
            edge = canny.detect_edge()
            # Generate result plots
            canny.plot_result()

            # Save plot
            plt.tight_layout()
            plt.savefig(f"edge_{filename.split('.')[0]}_{i}.png")
            plt.show()

if __name__ == "__main__":
    main()