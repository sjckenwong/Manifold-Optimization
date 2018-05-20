function d = distance(x, y)
    XtY = x'*y;
    cos_princ_angle = svd(XtY);
    square_d = sum(real(acos(cos_princ_angle)).^2);
    d = sqrt(square_d);
end