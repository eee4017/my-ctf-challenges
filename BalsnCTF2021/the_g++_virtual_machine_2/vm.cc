size = 5;

defun fast_pow(base, power, MOD) {
  result = 1;
  while (power > 0) {
    if (power % 2 == 1) {
      result = (result * base) % MOD;
    }
    power = power / 2;
    base = (base * base) % MOD;
  }
  return result;
}

defun setMatrix(v, the, size) {
  i = 0;
  j = 0;
  idx = 0;
  val = 0;
  while (i < size) {
    j = 0;
    while (j < size) {
      idx = (i * size) + j;
      val = fast_pow(the, idx, 97);
      SET_VECTOR(v, idx, val);
      j = j + 1;
    }
    i = i + 1;
  }
  return v;
}

defun sumMatrix(v) {
  ans = 0;
  i = 0;
  j = 0;
  idx = 0;
  while (i < size) {
    j = 0;
    while (j < size) {
      idx = (i * size) + j;
      ans = ans + v[idx];
      j = j + 1;
    }
    i = i + 1;
  }
  return ans;
}

defun innLoop(v1, v2, i, j) {
  sum = 0;
  k = 0;
  while (k < size) {
    sum = (sum + (v1[(i * size) + k] * v2[(k * size) + j])) % 31;
    //
    k = k + 1;
  }
  return sum;
}

// Matrix Multiplication
defun mm(v3, v1, v2) {
  i = 0;
  j = 0;
  k = 0;
  sum = 0;
  while (i < size) {
    j = 0;
    while (j < size) {
      sum = innLoop(v1, v2, i, j);
      SET_VECTOR(v3, (i * size) + j, sum);
      j = j + 1;
    }
    i = i + 1;
  }
  return v3;
}

defun generate_sbox(sb, the) {
  i = 0;
  j = 0;
  test = 0;
  while (i < 33) {
    test = fast_pow(the, i, 31);
    if (test < 25) {
      SET_VECTOR(sb, j, test);
      j = j + 1;
      //
    }
    //
    i = i + 1;
    //
  }
  SET_VECTOR(sb, 24, 0);

  return sb;
}

defun permute(v_new, v, sb) {
  i = 0;
  test = 0;
  while (i < 25) {
    test = v[i];
    SET_VECTOR(v_new, sb[i], test);
    //
    i = i + 1;
    //
  }
  //
  return v_new;
}

v1 = VECTOR();
SET_VECTOR(v1, 0, IN(0));
SET_VECTOR(v1, 1, IN(1));
SET_VECTOR(v1, 2, IN(2));
SET_VECTOR(v1, 3, IN(3));
SET_VECTOR(v1, 4, IN(4));
SET_VECTOR(v1, 5, IN(5));
SET_VECTOR(v1, 6, IN(6));
SET_VECTOR(v1, 7, IN(7));
SET_VECTOR(v1, 8, IN(8));
SET_VECTOR(v1, 9, IN(9));
SET_VECTOR(v1, 10, IN(10));
SET_VECTOR(v1, 11, IN(11));
SET_VECTOR(v1, 12, IN(12));
SET_VECTOR(v1, 13, IN(13));
SET_VECTOR(v1, 14, IN(14));
SET_VECTOR(v1, 15, IN(15));
SET_VECTOR(v1, 16, IN(16));
SET_VECTOR(v1, 17, IN(17));
SET_VECTOR(v1, 18, IN(18));
SET_VECTOR(v1, 19, IN(19));
SET_VECTOR(v1, 20, IN(20));
SET_VECTOR(v1, 21, IN(21));
SET_VECTOR(v1, 22, IN(22));
SET_VECTOR(v1, 23, IN(23));
SET_VECTOR(v1, 24, IN(24));

v2 = VECTOR();
v2 = setMatrix(v2, 5, size);

v3 = VECTOR();
v3 = mm(v3, v1, v2);

sb1 = VECTOR();
sb1 = generate_sbox(sb1, 3);

vnew = VECTOR();
vnew = permute(vnew, v3, sb1);

//

v1 = vnew;
v2 = setMatrix(v2, 15, size);

v3 = VECTOR();
v3 = mm(v3, v1, v2);

sb2 = VECTOR();
sb2 = generate_sbox(sb2, 11);

vnew = VECTOR();
vnew = permute(vnew, v3, sb2);

//

v1 = vnew;
v2 = setMatrix(v2, 27, size);
v3 = mm(v3, v1, v2);

sb3 = VECTOR();
sb3 = generate_sbox(sb3, 24);

vnew = VECTOR();
vnew = permute(vnew, v3, sb3);

//
//

vans = VECTOR();
SET_VECTOR(vans, 0, 28);
SET_VECTOR(vans, 1, 3);
SET_VECTOR(vans, 2, 26);
SET_VECTOR(vans, 3, 27);
SET_VECTOR(vans, 4, 15);
SET_VECTOR(vans, 5, 18);
SET_VECTOR(vans, 6, 13);
SET_VECTOR(vans, 7, 6);
SET_VECTOR(vans, 8, 2);
SET_VECTOR(vans, 9, 26);
SET_VECTOR(vans, 10, 25);
SET_VECTOR(vans, 11, 4);
SET_VECTOR(vans, 12, 10);
SET_VECTOR(vans, 13, 3);
SET_VECTOR(vans, 14, 17);
SET_VECTOR(vans, 15, 4);
SET_VECTOR(vans, 16, 29);
SET_VECTOR(vans, 17, 23);
SET_VECTOR(vans, 18, 17);
SET_VECTOR(vans, 19, 13);
SET_VECTOR(vans, 20, 7);
SET_VECTOR(vans, 21, 6);
SET_VECTOR(vans, 22, 15);
SET_VECTOR(vans, 23, 24);
SET_VECTOR(vans, 24, 22);


vfinal = vnew;

ans = 0;
i = 0;
j = 0;
idx = 0;
while (i < size) {
  j = 0;
  while (j < size) {
    idx = (i * size) + j;
    if(vfinal[idx] == vans[idx]){
      ans = ans + 1;
      //
    }
    //
    j = j + 1;
  }
  i = i + 1;
  //
}


OUT(ans);
