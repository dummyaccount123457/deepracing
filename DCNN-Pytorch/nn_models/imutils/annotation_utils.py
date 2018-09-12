def overlay_image(dest_image, src_image, offset=None ):
  #  print(src_image.shape)
  #  print(dest_image.shape)
    if offset is None:
        y_offset = int(round(dest_image.shape[0]/2.0 - src_image.shape[0]/2.0))
        x_offset = int(round(dest_image.shape[1]/2.0 - src_image.shape[1]/2.0))
    else:
        x_offset = offset[0]
        y_offset = offset[1]
    rtn = dest_image.copy()
    y1, y2 = y_offset, y_offset + src_image.shape[0]
    x1, x2 = x_offset, x_offset + src_image.shape[1]

    alpha_s = src_image[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        rtn[y1:y2, x1:x2, c] = (alpha_s * src_image[:, :, c] +
                                  alpha_l * rtn[y1:y2, x1:x2, c])
    return rtn