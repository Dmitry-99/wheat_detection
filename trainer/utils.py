# this function will take the dataframe and vertically stack the image ids 
# with no bounding boxes
def process_bbox(df):
    df['bbox'] = df['bbox'].apply(lambda x: eval(x))
    df['x'] = df['bbox'].apply(lambda x: x[0])
    df['y'] = df['bbox'].apply(lambda x: x[1])
    df['w'] = df['bbox'].apply(lambda x: x[2])
    df['h'] = df['bbox'].apply(lambda x: x[3])
    df['x'] = df['x'].astype(float)
    df['y'] = df['y'].astype(float)
    df['w'] = df['w'].astype(float)
    df['h'] = df['h'].astype(float)

    df.drop(columns=['bbox'],inplace=True)
    return df

def collate_fn(batch):
    return tuple(zip(*batch))

def image_convert(image):
    image = image.clone().cpu().numpy()
    image = image.transpose((1,2,0))
    image = (image * 255).astype(np.uint8)
    return image

def format_prediction_string(boxes, scores):
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))
    return " ".join(pred_strings)