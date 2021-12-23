
// defun fast_pow(base, power, MOD) {
//   result = 1;
//   while (power > 0) {
//     if (power % 2 == 1) {
//       result = (((result % MOD) * (base % MOD)) % MOD);
//     }
//     power = power / 2;
//     base = (((base % MOD) * (base % MOD)) % MOD);
//   }
//   return result;
// }
// OUT(fast_pow(IN(0), 492853, 130253));


v1 = VECTOR();

SET_VECTOR(v1, 0, IN(0));
SET_VECTOR(v1, 1, IN(1));
SET_VECTOR(v1, 2, IN(2));
SET_VECTOR(v1, 3, IN(3));
SET_VECTOR(v1, 4, IN(4));
SET_VECTOR(v1, 5, IN(5));

h = VECTOR();
SET_VECTOR(h, 0, 1);
SET_VECTOR(h, 1, 26);
SET_VECTOR(h, 2, 676);
SET_VECTOR(h, 3, 17576);
SET_VECTOR(h, 4, 456976);
SET_VECTOR(h, 5, 11881376);

defun hash(v, h){
  i = 0;
  ans = 0;
  while(i < 5){
    ans = ans + ( ((v1[i] - 97) % 26) * h[i]);
    ans = ans % 488148229;
    i = i + 1;
  }
  return ans;
}

OUT(hash(v1, h));


