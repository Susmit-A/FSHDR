function B = LDRtoLDR_Sen(A, expA, expB)
Radiance = LDRtoHDR_Sen(A, expA);
B = HDRtoLDR_Sen(Radiance, expB);