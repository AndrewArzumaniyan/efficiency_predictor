
      program INTRINSICS
      print *, '=== START OF F2C_MATH intrinsic test ==========='

c     TESTING abs SPECIFIC INTRINSIC
c     integer*1 babs(integer*1)
      call abs9
c     integer*2 iiabs(integer*2)
      call abs10
c     integer*2 habs(integer*2)
      call abs11
c     integer*4 jiabs(integer*4)
      call abs12
c     integer*8 kiabs(integer*8)
      call abs13

c     TESTING acosd GENERIC INTRINSIC
c     real*4 acosd(real*4)
      call acosd1
c     real*8 acosd(real*8)
      call acosd2
c     real*8 dacosd(real*8)
      call acosd3

c     TESTING asind GENERIC INTRINSIC
c     real*4 asind(real*4)
      call asind1
c     real*8 asind(real*8)
      call asind2
c     real*8 dasind(real*8)
      call asind3

c     TESTING atand GENERIC INTRINSIC
c     real*4 atand(real*4)
      call atand1
c     real*8 atand(real*8)
      call atand2
c     real*8 datand(real*8)
      call atand3

c     TESTING atan2d GENERIC INTRINSIC
c     real*4 atan2d(real*4, real*4)
      call atan2d1
c     real*8 atan2d(real*8, real*8)
      call atan2d2
c     real*8 datan2d(real*8, real*8)
      call atan2d3

c     TESTING btest SPECIFIC INTRINSIC
c     logical*1 bbtest(integer*1)
      call btest5
c     logical*2 bitest(integer*2)
      call btest6
c     logical*2 htest(integer*2)
      call btest7
c     logical*4 bjtest(integer*4)
      call btest8
c     logical*8 bktest(integer*8)
      call btest9

c     TESTING cosd GENERIC INTRINSIC
c     real*4 cosd(real*4)
      call cosd1
c     real*8 cosd(real*8)
      call cosd2
c     real*8 dcosd(real*8)
      call cosd3

c     TESTING cotan GENERIC INTRINSIC
c     real*4 cotan(real*4)
      call cotan1
c     real*8 cotan(real*8)
      call cotan2
c     real*8 dcotan(real*8)
      call cotan3

c     TESTING cotand GENERIC INTRINSIC
c     real*4 cotand(real*4)
      call cotand1
c     real*8 cotand(real*8)
      call cotand2
c     real*8 dcotand(real*8)
      call cotand3

c     TESTING dfloat SPECIFIC INTRINSIC
c     real*8 dfloti(integer*2)
      call dfloat5
c     real*8 dflotj(integer*4)
      call dfloat6
c     real*8 dflotk(integer*8)
      call dfloat7

c     TESTING dim SPECIFIC INTRINSIC
c     integer*1 bdim(integer*1)
      call dim7
c     integer*2 iidim(integer*2)
      call dim8
c     integer*2 hdim(integer*2)
      call dim9
c     integer*4 idim(integer*4)
      call dim10
c     integer*4 jidim(integer*4)
      call dim11
c     integer*8 kidim(integer*8)
      call dim12

c     TESTING iand SPECIFIC INTRINSIC
c     integer*1 biand(integer*1)
      call iand9
c     integer*2 iiand(integer*2)
      call iand10
c     integer*2 hiand(integer*2)
      call iand11
c     integer*4 jiand(integer*4)
      call iand12
c     integer*8 kiand(integer*8)
      call iand13

c     TESTING ibchng GENERIC INTRINSIC
c     integer*1 ibchng(integer*1)
      call ibchng1
c     integer*2 ibchng(integer*2)
      call ibchng2
c     integer*4 ibchng(integer*4)
      call ibchng3
c     integer*8 ibchng(integer*8)
      call ibchng4

c     TESTING ibclr SPECIFIC INTRINSIC
c     integer*1 bbclr(integer*1)
      call ibclr5
c     integer*2 iibclr(integer*2)
      call ibclr6
c     integer*2 hbclr(integer*2)
      call ibclr7
c     integer*4 jibclr(integer*4)
      call ibclr8
c     integer*8 kibclr(integer*8)
      call ibclr9

c     TESTING ibits SPECIFIC INTRINSIC
c     integer*1 bbits(integer*1)
      call ibits5
c     integer*2 iibits(integer*2)
      call ibits6
c     integer*2 hbits(integer*2)
      call ibits7
c     integer*4 jibits(integer*4)
      call ibits8
c     integer*8 kibits(integer*8)
      call ibits9

c     TESTING ibset SPECIFIC INTRINSIC
c     integer*1 bbset(integer*1)
      call ibset5
c     integer*2 iibset(integer*2)
      call ibset6
c     integer*2 hbset(integer*2)
      call ibset7
c     integer*4 jibset(integer*4)
      call ibset8
c     integer*8 kibset(integer*8)
      call ibset9

c     TESTING ieor GENERIC INTRINSIC
c     integer*1 ixor(integer*1)
      call ieor5
c     integer*2 ixor(integer*2)
      call ieor6
c     integer*4 ixor(integer*4)
      call ieor7
c     integer*8 ixor(integer*8)
      call ieor8
c     integer*1 bieor(integer*1)
      call ieor13
c     integer*1 bixor(integer*1)
      call ieor14
c     integer*2 iieor(integer*2)
      call ieor15
c     integer*2 hieor(integer*2)
      call ieor16
c     integer*2 iixor(integer*2)
      call ieor17
c     integer*2 hixor(integer*2)
      call ieor18
c     integer*4 jieor(integer*4)
      call ieor19
c     integer*4 jixor(integer*4)
      call ieor20
c     integer*8 kieor(integer*8)
      call ieor21

c     TESTING ilen GENERIC INTRINSIC
c     integer*1 ilen(integer*1)
      call ilen1
c     integer*2 ilen(integer*2)
      call ilen2
c     integer*4 ilen(integer*4)
      call ilen3
c     integer*8 ilen(integer*8)
      call ilen4

c     TESTING ior SPECIFIC INTRINSIC
c     integer*1 bior(integer*1)
      call ior9
c     integer*2 iior(integer*2)
      call ior10
c     integer*2 hior(integer*2)
      call ior11
c     integer*4 jior(integer*4)
      call ior12
c     integer*8 kior(integer*8)
      call ior13

c     TESTING isha GENERIC INTRINSIC
c     integer*1 isha(integer*1)
      call isha1
c     integer*2 isha(integer*2)
      call isha2
c     integer*4 isha(integer*4)
      call isha3
c     integer*8 isha(integer*8)
      call isha4

c     TESTING ishc GENERIC INTRINSIC
c     integer*1 ishc(integer*1)
      call ishc1
c     integer*2 ishc(integer*2)
      call ishc2
c     integer*4 ishc(integer*4)
      call ishc3
c     integer*8 ishc(integer*8)
      call ishc4

c     TESTING ishft GENERIC INTRINSIC
c     integer*1 bshft(integer*1)
      call ishft5
c     integer*2 iishft(integer*2)
      call ishft6
c     integer*2 hshft(integer*2)
      call ishft7
c     integer*4 jishft(integer*4)
      call ishft8
c     integer*8 kishft(integer*8)
      call ishft9

c     TESTING lshft GENERIC INTRINSIC
c     integer*1 lshft(integer*1)
      call lshft1
c     integer*2 lshft(integer*2)
      call lshft2
c     integer*4 lshft(integer*4)
      call lshft3
c     integer*8 lshft(integer*8)
      call lshft4

c     TESTING rshft GENERIC INTRINSIC
c     integer*1 rshft(integer*1)
      call rshft1
c     integer*2 rshft(integer*2)
      call rshft2
c     integer*4 rshft(integer*4)
      call rshft3
c     integer*8 rshft(integer*8)
      call rshft4

c     TESTING ishftc SPECIFIC INTRINSIC
c     integer*1 bshftc(integer*1)
      call ishftc9
c     integer*1 bshftc(integer*1)
      call ishftc10
c     integer*2 iishftc(integer*2)
      call ishftc11
c     integer*2 iishftc(integer*2)
      call ishftc12
c     integer*2 hshftc(integer*2)
      call ishftc13
c     integer*2 hshftc(integer*2)
      call ishftc14
c     integer*4 jishftc(integer*4)
      call ishftc15
c     integer*4 jishftc(integer*4)
      call ishftc16
c     integer*8 kishftc(integer*8)
      call ishftc17
c     integer*8 kishftc(integer*8)
      call ishftc18

c     TESTING ishl GENERIC INTRINSIC
c     integer*1 ishl(integer*1)
      call ishl1
c     integer*2 ishl(integer*2)
      call ishl2
c     integer*4 ishl(integer*4)
      call ishl3
c     integer*8 ishl(integer*8)
      call ishl4
                                    
c     TESTING log10 GENERIC INTRINSIC
c     complex*8 log10(complex*8)
      call log103
c     complex*16 log10(complex*16)
      call log104 
c     complex*8 clog10(complex*8)
      call log107
c     complex*16 cdlog10(complex*16)
      call log108

c     TESTING max SPECIFIC INTRINSIC
c     integer*2 imax0(integer*2)
      call max10_
c     integer*4 jmax0(integer*4)
      call max11_
c     integer*8 kmax0(integer*8)
      call max12_
c     integer*2 imax1(real*4)
      call max14_
c     integer*4 jmax1(real*4)
      call max15_
c     integer*8 kmax1(real*4)
      call max16_
c     real*4 aimax0(integer*2)
      call max18_
c     real*4 ajmax0(integer*4)
      call max19_
c     real*4 akmax0(integer*8)
      call max20_

c     TESTING min SPECIFIC INTRINSIC
c     integer*2 imin0(integer*2)
      call min10_
c     integer*4 jmin0(integer*4)
      call min11_
c     integer*8 kmin0(integer*8)
      call min12_

c     integer*2 imin1(real*4)
      call min14_
c     integer*4 jmin1(real*4)
      call min15_
c     integer*8 kmin1(real*4)
      call min16_
c     real*4 aimin0(integer*2)

      call min18_
c     real*4 ajmin0(integer*4)
      call min19_
c     real*4 akmin0(integer*8)
      call min20_

c     TESTING mod SPECIFIC INTRINSIC
c     integer*1 bmod(integer*1)
      call mod5
c     integer*2 imod(integer*2)
      call mod6
c     integer*2 hmod(integer*2)
      call mod7
c     integer*4 jmod(integer*4)
      call mod8
c     integer*8 kmod(integer*8)
      call mod9

c     TESTING not SPECIFIC INTRINSIC
c     integer*1 bnot(integer*1)
      call not5
c     integer*2 inot(integer*2)
      call not6
c     integer*2 hnot(integer*2)
      call not7
c     integer*4 jnot(integer*4)
      call not8
c     integer*8 knot(integer*8)
      call not9

c     TESTING isign SPECIFIC INTRINSIC
c     integer*1 isign(integer*1)
      call sign7
c     integer*2 isign(integer*2)
      call sign8
c     integer*8 isign(integer*8)
      call sign10
c     integer*1 bsign(integer*1)
      call sign11
c     integer*2 iisign(integer*2)
      call sign12
c     integer*2 hsign(integer*2)
      call sign13
c     integer*4 jisign(integer*4)
      call sign14
c     integer*8 kisign(integer*8)
      call sign15

c     TESTING float SPECIFIC INTRINSIC
c     real*4 floati(integer*2)
      call real9
c     real*4 floatj(integer*4)
      call real11
c     real*4 floatk(integer*8)
      call real12


c     TESTING tan SPECIFIC INTRINSIC
c     complex*8 ctan(complex*8)
      call tan6
c     complex*16 cdtan(complex*16)
      call tan7
c     complex*16 ztan(complex*16)
      call tan8

c     TESTING sind SPECIFIC INTRINSIC
c     real*4 sind(real*4)
      call sind1
c     real*8 sind(real*8)
      call sind2
c     real*8 dsind(real*8)
      call sind3
                       
c     TESTING tand GENERIC INTRINSIC
c     real*4 tand(real*4)
      call tand1
c     real*8 tand(real*8)
      call tand2
c     real*8 dtand(real*8)
      call tand3


      print *, '=== END OF F2C_MATH intrinsic test ============='
      end

C -------------------------------------------------

      subroutine abs9
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*1 A(N), B(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'babs_char'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B)	  
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = babs(B(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (babs(B(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif
      end

C -------------------------------------------------

      subroutine abs10
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*2 A(N), B(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'iiabs_short'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B)	  
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = iiabs(B(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (iiabs(B(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif
      end

C -------------------------------------------------

      subroutine abs11
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*2 A(N), B(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'habs_short'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B)	  
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = habs(B(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (habs(B(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif
      end

C -------------------------------------------------

      subroutine abs12
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer A(N), B(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'jiabs_long'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B)	  
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = jiabs(B(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (jiabs(B(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif
      end

C -------------------------------------------------

      subroutine abs13
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*8 A(N), B(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'kiabs_longlong'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B)	  
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = kiabs(B(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (kiabs(B(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif
      end

C -------------------------------------------------

      subroutine acosd1
      integer, parameter :: N = 256, ER = N + 1, W = 2, S = -1
      real, parameter :: EPS = 1e-6
      character*24 tname
      real A(N), B(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'acosd_float'
      erri = ER

      do i = 1, N
          call random_number(B(i))
          B(i) = B(i) * W + S
      enddo

!dvm$ actual(B)	  
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = acosd(B(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri)), private(tmp)
      do i = 1, N
          tmp = acosd(B(i))
          if (abs(tmp - A(i))/abs(tmp) .gt. EPS) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif
      end

C -------------------------------------------------

      subroutine acosd2
      integer, parameter :: N = 256, ER = N + 1, W = 2, S = -1
      real*8, parameter :: EPS = 1d-15
      character*24 tname
      real*8 A(N), B(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real*8 tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'acosd_double'
      erri = ER

      do i = 1, N
          call random_number(B(i))
          B(i) = B(i) * W + S
      enddo

!dvm$ actual(B)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = acosd(B(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri)), private(tmp)
      do i = 1, N
          tmp = acosd(B(i))
          if (abs(tmp - A(i))/abs(tmp) .gt. EPS) then
             erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif
      end

C -------------------------------------------------

      subroutine acosd3
      integer, parameter :: N = 256, ER = N + 1, W = 2, S = -1
      real*8, parameter :: EPS = 1d-15
      character*24 tname
      real*8 A(N), B(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real*8 tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'dacosd_double'
      erri = ER

      do i = 1, N
          call random_number(B(i))
          B(i) = B(i) * W + S
      enddo

!dvm$ actual(B)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = dacosd(B(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri)), private(tmp)
      do i = 1, N
          tmp = dacosd(B(i))
          if (abs(tmp - A(i))/abs(tmp) .gt. EPS) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif
      end

C -------------------------------------------------

      subroutine asind1
      integer, parameter :: N = 256, ER = N + 1, W = 2, S = -1
      real, parameter :: EPS = 1e-6
      character*24 tname
      real A(N), B(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'asind_float'
      erri = ER

      do i = 1, N
          call random_number(B(i))
          B(i) = B(i) * W + S
      enddo

!dvm$ actual(B)	  
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = asind(B(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri)), private(tmp)
      do i = 1, N
          tmp = asind(B(i))
          if (abs(tmp - A(i))/abs(tmp) .gt. EPS) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif
      end

C -------------------------------------------------

      subroutine asind2
      integer, parameter :: N = 256, ER = N + 1, W = 2, S = -1
      real*8, parameter :: EPS = 1d-15
      character*24 tname
      real*8 A(N), B(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real*8 tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'asind_double'
      erri = ER

      do i = 1, N
          call random_number(B(i))
          B(i) = B(i) * W + S
      enddo

!dvm$ actual(B)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = asind(B(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri)), private(tmp)
      do i = 1, N
          tmp = asind(B(i))
          if (abs(tmp - A(i))/abs(tmp) .gt. EPS) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif
      end

C -------------------------------------------------

      subroutine asind3
      integer, parameter :: N = 256, ER = N + 1, W = 2, S = -1
      real*8, parameter :: EPS = 1d-15
      character*24 tname
      real*8 A(N), B(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real*8 tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'dasind_double'
      erri = ER

      do i = 1, N
          call random_number(B(i))
          B(i) = B(i) * W + S
      enddo

!dvm$ actual(B)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = dasind(B(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri)), private(tmp)
      do i = 1, N
          tmp = dasind(B(i))
          if (abs(tmp - A(i))/abs(tmp) .gt. EPS) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif
      end
C -------------------------------------------------

      subroutine atand1
      integer, parameter :: N = 256, ER = N + 1, W = 100, S = 0
      real, parameter :: EPS = 1e-6
      character*24 tname
      real A(N), B(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'atand_float'
      erri = ER

      do i = 1, N
          call random_number(B(i))
          B(i) = B(i) * W + S
      enddo

!dvm$ actual(B)	  
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = atand(B(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri)), private(tmp)
      do i = 1, N
          tmp = atand(B(i))
          if (abs(tmp - A(i))/abs(tmp) .gt. EPS) then
               erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif
      end

C -------------------------------------------------

      subroutine atand2
      integer, parameter :: N = 256, ER = N + 1, W = 100, S = 0
      real*8, parameter :: EPS = 1d-15
      character*24 tname
      real*8 A(N), B(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real*8 tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'atand_double'
      erri = ER

      do i = 1, N
          call random_number(B(i))
          B(i) = B(i) * W + S
      enddo

!dvm$ actual(B)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = atand(B(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri)), private(tmp)
      do i = 1, N
          tmp = atand(B(i))
          if (abs(tmp - A(i))/abs(tmp) .gt. EPS) then
               erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine atand3
      integer, parameter :: N = 256, ER = N + 1, W = 100, S = 0
      real*8, parameter :: EPS = 1d-15
      character*24 tname
      real*8 A(N), B(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real*8 tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'datand_double'
      erri = ER

      do i = 1, N
          call random_number(B(i))
          B(i) = B(i) * W + S
      enddo

!dvm$ actual(B)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = datand(B(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri)), private(tmp)
      do i = 1, N
          tmp = datand(B(i))
          if (abs(tmp - A(i))/abs(tmp) .gt. EPS) then
               erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine atan2d1
      integer, parameter :: N = 256, ER = N + 1, W1 = 200, S1 = -100, W2
     & = 99, S2 = 1
      real, parameter :: EPS = 1e-6
      character*24 tname
      real A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'atan2d_float'
      erri = ER

      do i = 1, N
          call random_number(B(i))
          call random_number(C(i))
          B(i) = B(i) * W1 + S1
          C(i) = C(i) * W2 + S2
      enddo

!dvm$ actual(B, C)	  
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = atan2d(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri)), private(tmp)
      do i = 1, N
          tmp = atan2d(B(i), C(i))
          if (abs(tmp - A(i))/abs(tmp) .gt. EPS) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif
      end

C -------------------------------------------------

      subroutine atan2d2
      integer, parameter :: N = 256, ER = N + 1, W1 = 200, S1 = -100, W2
     & = 99, S2 = 1
      real*8, parameter :: EPS = 1d-15
      character*24 tname
      real*8 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real*8 tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'atan2d_double'
      erri = ER

      do i = 1, N
          call random_number(B(i))
          call random_number(C(i))
          B(i) = B(i) * W1 + S1
          C(i) = C(i) * W2 + S2
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = atan2d(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri)), private(tmp)
      do i = 1, N
          tmp = atan2d(B(i), C(i))
          if (abs(tmp - A(i))/abs(tmp) .gt. EPS) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif
      end

C -------------------------------------------------

      subroutine atan2d3
      integer, parameter :: N = 256, ER = N + 1, W1 = 200, S1 = -100, W2
     & = 99, S2 = 1
      real*8, parameter :: EPS = 1d-15
      character*24 tname
      real*8 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real*8 tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'datan2d_double'
      erri = ER

      do i = 1, N
          call random_number(B(i))
          call random_number(C(i))
          B(i) = B(i) * W1 + S1
          C(i) = C(i) * W2 + S2
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = datan2d(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri)), private(tmp)
      do i = 1, N
          tmp = datan2d(B(i), C(i))
          if (abs(tmp - A(i))/abs(tmp) .gt. EPS) then
             erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif
      end

C -------------------------------------------------

      subroutine btest5
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      logical*1 A(N)
      integer*1 B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'bbtest_char'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * bit_size(C(i)))
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = bbtest(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (bbtest(B(i), C(i)) .neqv. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif
      end

C -------------------------------------------------

      subroutine btest6
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      logical*2 A(N)
      integer*2 B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'bitest_short'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * bit_size(C(i)))
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = bitest(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (bitest(B(i), C(i)) .neqv. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif
      end

C -------------------------------------------------

      subroutine btest7
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      logical*2 A(N)
      integer*2 B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'htest_short'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * bit_size(C(i)))
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = htest(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (htest(B(i), C(i)) .neqv. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif
      end

C -------------------------------------------------

      subroutine btest8
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      logical*4 A(N)
      integer*4 B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'bjtest_long'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * bit_size(C(i)))
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = bjtest(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (bjtest(B(i), C(i)) .neqv. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif
      end

C -------------------------------------------------

      subroutine btest9
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      logical*8 A(N)
      integer*8 B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'bktest_longlong'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * bit_size(C(i)))
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = bktest(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (bktest(B(i), C(i)) .neqv. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif
      end

C -------------------------------------------------

      subroutine cosd1
      integer, parameter :: N = 256, ER = N + 1, W = 720, S = -360
      real, parameter :: EPS = 1e-6
      character*24 tname
      real A(N), B(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'cosd_float'
      erri = ER

      do i = 1, N
          call random_number(B(i))
          B(i) = B(i) * W + S
      enddo

!dvm$ actual(B)	  
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = cosd(B(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri)), private(tmp)
      do i = 1, N
          tmp = cosd(B(i))
          if (abs(tmp - A(i))/abs(tmp) .gt. EPS) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine cosd2
      integer, parameter :: N = 256, ER = N + 1, W = 720, S = -360
      real*8, parameter :: EPS = 1d-11
      character*24 tname
      real*8 A(N), B(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real*8 tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'cosd_double'
      erri = ER

      do i = 1, N
          call random_number(B(i))
          B(i) = B(i) * W + S
      enddo

!dvm$ actual(B)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = cosd(B(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri)), private(tmp)
      do i = 1, N
          tmp = cosd(B(i))
          if (abs(tmp - A(i))/abs(tmp) .gt. EPS) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine cosd3
      integer, parameter :: N = 256, ER = N + 1, W = 720, S = -360
      real*8, parameter :: EPS = 1d-11
      character*24 tname
      real*8 A(N), B(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real*8 tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'dcosd_double'
      erri = ER

      do i = 1, N
          call random_number(B(i))
          B(i) = B(i) * W + S
      enddo

!dvm$ actual(B)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = dcosd(B(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri)), private(tmp)
      do i = 1, N
          tmp = dcosd(B(i))
          if (abs(tmp - A(i))/abs(tmp) .gt. EPS) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine cotan1
      integer, parameter :: N = 256, ER = N + 1, W = 99, S = 1
      real, parameter :: EPS = 1e-6
      character*24 tname
      real A(N), B(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'cotan_float'
      erri = ER

      do i = 1, N
          call random_number(B(i))
          B(i) = B(i) * W + S
      enddo

!dvm$ actual(B)	  
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = cotan(B(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri)), private(tmp)
      do i = 1, N
          tmp = cotan(B(i))
          if (abs(tmp - A(i))/abs(tmp) .gt. EPS) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine cotan2
      integer, parameter :: N = 256, ER = N + 1, W = 99, S = 1
      real*8, parameter :: EPS = 1d-15
      character*24 tname
      real*8 A(N), B(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real*8 tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'cotan_double'
      erri = ER

      do i = 1, N
          call random_number(B(i))
          B(i) = B(i) * W + S
      enddo

!dvm$ actual(B)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = cotan(B(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri)), private(tmp)
      do i = 1, N
          tmp = cotan(B(i))
          if (abs(tmp - A(i))/abs(tmp) .gt. EPS) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine cotan3
      integer, parameter :: N = 256, ER = N + 1, W = 99, S = 1
      real*8, parameter :: EPS = 1d-11
      character*24 tname
      real*8 A(N), B(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real*8 tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'dcotan_double'
      erri = ER

      do i = 1, N
          call random_number(B(i))
          B(i) = B(i) * W + S
      enddo

!dvm$ actual(B)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = dcotan(B(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri)), private(tmp)
      do i = 1, N
          tmp = dcotan(B(i))
          if (abs(tmp - A(i))/abs(tmp) .gt. EPS) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine cotand1
      integer, parameter :: N = 256, ER = N + 1, W = 120, S = 30
      real, parameter :: EPS = 1e-6
      character*24 tname
      real A(N), B(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'cotand_float'
      erri = ER

      do i = 1, N
          call random_number(B(i))
          B(i) = B(i) * W + S
      enddo

!dvm$ actual(B)	  
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = cotand(B(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri)), private(tmp)
      do i = 1, N
          tmp = cotand(B(i))
          if (abs(tmp - A(i))/abs(tmp) .gt. EPS) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine cotand2
      integer, parameter :: N = 256, ER = N + 1, W = 120, S = 30
      real*8, parameter :: EPS = 1d-11
      character*24 tname
      real*8 A(N), B(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real*8 tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'cotand_double'
      erri = ER

      do i = 1, N
          call random_number(B(i))
          B(i) = B(i) * W + S
      enddo

!dvm$ actual(B)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = cotand(B(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri)), private(tmp)
      do i = 1, N
          tmp = cotand(B(i))
          if (abs(tmp - A(i))/abs(tmp) .gt. EPS) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine cotand3
      integer, parameter :: N = 256, ER = N + 1, W = 120, S = 30
      real*8, parameter :: EPS = 1d-12
      character*24 tname
      real*8 A(N), B(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real*8 tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'dcotand_double'
      erri = ER

      do i = 1, N
          call random_number(B(i))
          B(i) = B(i) * W + S
      enddo

!dvm$ actual(B)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = dcotand(B(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri)), private(tmp)
      do i = 1, N
          tmp = dcotand(B(i))
          if (abs(tmp - A(i))/abs(tmp) .gt. EPS) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine dfloat5
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      real*8 A(N)
      integer*2 B(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'dfloti_short'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = dfloti(B(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (dfloti(B(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine dfloat6
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      real*8 A(N)
      integer*4 B(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'dflotj_long'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = dflotj(B(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (dflotj(B(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine dfloat7
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      real*8 A(N)
      integer*8 B(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'dflotk_longlong'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = dflotk(B(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (dflotk(B(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine dim7
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*1 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'bdim_char'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = bdim(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (bdim(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine dim8
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*2 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'iidim_short'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = iidim(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (iidim(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine dim9
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*2 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'hdim_short'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = hdim(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (hdim(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine dim10
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*4 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'idim_long'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = idim(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (idim(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine dim11
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*4 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'jidim_long'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = jidim(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (jidim(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine dim12
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*8 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'kidim_longlong'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = kidim(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (kidim(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine iand9
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*1 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'biand_char'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = biand(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (biand(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine iand10
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*2 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'iiand_short'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = iiand(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (iiand(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine iand11
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*2 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'hiand_short'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = hiand(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (hiand(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine iand12
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*4 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'jiand_long'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = jiand(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (jiand(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine iand13
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*8 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'kiand_longlong'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = kiand(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (kiand(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ibchng1
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*1 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'ibchng_char'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * bit_size(C(i)))
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = ibchng(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (ibchng(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif
      end

C -------------------------------------------------

      subroutine ibchng2
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*2 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'ibchng_short'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * bit_size(C(i)))
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = ibchng(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (ibchng(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif
      end

C -------------------------------------------------

      subroutine ibchng3
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*4 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'ibchng_long'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * bit_size(C(i)))
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = ibchng(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (ibchng(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif
      end

C -------------------------------------------------

      subroutine ibchng4
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*8 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'ibchng_longlong'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * bit_size(C(i)))
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = ibchng(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (ibchng(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif
      end

C -------------------------------------------------

      subroutine ibclr5
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*1 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'bbclr_char'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * bit_size(C(i)))
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = bbclr(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (bbclr(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ibclr6
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*2 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'iibclr_short'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * bit_size(C(i)))
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = iibclr(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (iibclr(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ibclr7
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*2 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'hbclr_short'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * bit_size(C(i)))
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = hbclr(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (hbclr(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ibclr8
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*4 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'jibclr_long'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * bit_size(C(i)))
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = jibclr(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (jibclr(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ibclr9
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*8 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'kibclr_longlong'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * bit_size(C(i)))
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = kibclr(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (kibclr(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ibits5
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*1 A(N), B(N), C(N), D(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'bbits_char'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          D(i) = int(tmp * (bit_size(D(i)) + 1))
          call random_number(tmp)
          C(i) = int(tmp * (bit_size(C(i)) + 1 - D(i)))
      enddo

!dvm$ actual(B, C, D)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = bbits(B(i), C(i), D(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (bbits(B(i), C(i), D(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ibits6
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*2 A(N), B(N), C(N), D(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'iibits_short'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          D(i) = int(tmp * (bit_size(D(i)) + 1))
          call random_number(tmp)
          C(i) = int(tmp * (bit_size(C(i)) + 1 - D(i)))
      enddo

!dvm$ actual(B, C, D)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = iibits(B(i), C(i), D(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (iibits(B(i), C(i), D(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ibits7
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*2 A(N), B(N), C(N), D(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'hbits_short'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          D(i) = int(tmp * (bit_size(D(i)) + 1))
          call random_number(tmp)
          C(i) = int(tmp * (bit_size(C(i)) + 1 - D(i)))
      enddo

!dvm$ actual(B, C, D)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = hbits(B(i), C(i), D(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (hbits(B(i), C(i), D(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ibits8
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*4 A(N), B(N), C(N), D(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'jibits_long'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          D(i) = int(tmp * (bit_size(D(i)) + 1))
          call random_number(tmp)
          C(i) = int(tmp * (bit_size(C(i)) + 1 - D(i)))
      enddo

!dvm$ actual(B, C, D)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = jibits(B(i), C(i), D(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (jibits(B(i), C(i), D(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ibits9
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*8 A(N), B(N), C(N), D(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'kibits_longlong'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          D(i) = int(tmp * (bit_size(D(i)) + 1))
          call random_number(tmp)
          C(i) = int(tmp * (bit_size(C(i)) + 1 - D(i)))
      enddo

!dvm$ actual(B, C, D)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = kibits(B(i), C(i), D(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (kibits(B(i), C(i), D(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ibset5
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*1 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'bbset_char'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * bit_size(C(i)))
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = bbset(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (bbset(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ibset6
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*2 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'iibset_short'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * bit_size(C(i)))
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = iibset(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (iibset(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ibset7
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*2 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'hbset_short'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * bit_size(C(i)))
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = hbset(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (hbset(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ibset8
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*4 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'jibset_long'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * bit_size(C(i)))
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = jibset(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (jibset(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ibset9
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*8 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'kibset_longlong'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * bit_size(C(i)))
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = kibset(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (kibset(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ieor5
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*1 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'ixor_char'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = ixor(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (ixor(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ieor6
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*2 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'ixor_short'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = ixor(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (ixor(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ieor7
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*4 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'ixor_long'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = ixor(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (ixor(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ieor8
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*8 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'ixor_longlong'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = ixor(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (ixor(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ieor13
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*1 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'bieor_char'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = bieor(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (bieor(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ieor14
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*1 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'bixor_char'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = bixor(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (bixor(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ieor15
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*2 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'iieor_short'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = iieor(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (iieor(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ieor16
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*2 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'hieor_short'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = hieor(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (hieor(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ieor17
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*2 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'iixor_short'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = iixor(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (iixor(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ieor18
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*2 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'hixor_short'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = hixor(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (hixor(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ieor19
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*4 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'jieor_long'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = jieor(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (jieor(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ieor20
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*4 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'jixor_long'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = jixor(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (jixor(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ieor21
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*8 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'kieor_longlong'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = kieor(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (kieor(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ilen1
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*1 A(N), B(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'ilen_char'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = ilen(B(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (ilen(B(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ilen2
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*2 A(N), B(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'ilen_short'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = ilen(B(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (ilen(B(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ilen3
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*4 A(N), B(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'ilen_long'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = ilen(B(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (ilen(B(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ilen4
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*8 A(N), B(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'ilen_longlong'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = ilen(B(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (ilen(B(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ior9
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*1 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp
	  
!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'bior_char'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = bior(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (bior(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ior10
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*2 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp
	  
!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'iior_short'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = iior(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (iior(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ior11
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*2 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp
	  
!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'hior_short'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = hior(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (hior(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ior12
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*4 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp
	  
!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'jior_long'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = jior(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (jior(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ior13
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*8 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp
	  
!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'kior_longlong'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = kior(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (kior(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine isha1
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*1 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'isha_char'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * (2 * bit_size(C(i)) + 1) - bit_size(C(i)))
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = isha(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (isha(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine isha2
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*2 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'isha_short'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * (2 * bit_size(C(i)) + 1) - bit_size(C(i)))
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = isha(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (isha(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine isha3
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*4 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'isha_long'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * (2 * bit_size(C(i)) + 1) - bit_size(C(i)))
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = isha(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (isha(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine isha4
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*8 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'isha_longlong'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * (2 * bit_size(C(i)) + 1) - bit_size(C(i)))
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = isha(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (isha(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ishc1
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*1 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'ishc_char'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * (2 * bit_size(C(i)) + 1) - bit_size(C(i)))
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = ishc(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (ishc(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ishc2
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*2 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'ishc_short'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * (2 * bit_size(C(i)) + 1) - bit_size(C(i)))
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = ishc(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (ishc(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ishc3
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*4 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'ishc_long'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * (2 * bit_size(C(i)) + 1) - bit_size(C(i)))
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = ishc(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (ishc(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ishc4
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*8 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'ishc_longlong'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * (2 * bit_size(C(i)) + 1) - bit_size(C(i)))
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = ishc(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (ishc(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ishft5
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*1 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'bshft_char'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * (2 * bit_size(C(i)) + 1) - bit_size(C(i)))
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = bshft(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (bshft(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ishft6
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*2 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'iishft_short'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * (2 * bit_size(C(i)) + 1) - bit_size(C(i)))
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = iishft(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (iishft(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ishft7
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*2 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'hshft_short'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * (2 * bit_size(C(i)) + 1) - bit_size(C(i)))
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = hshft(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (hshft(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ishft8
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*4 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'jishft_long'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * (2 * bit_size(C(i)) + 1) - bit_size(C(i)))
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = jishft(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (jishft(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ishft9
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*8 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'kishft_longlong'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * (2 * bit_size(C(i)) + 1) - bit_size(C(i)))
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = kishft(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (kishft(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine lshft1
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*1 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'lshft_char'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * (bit_size(C(i)) + 1))
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = lshft(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (lshft(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine lshft2
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*2 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'lshft_short'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * (bit_size(C(i)) + 1))
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = lshft(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (lshft(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine lshft3
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*4 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'lshft_long'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * (bit_size(C(i)) + 1))
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = lshft(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (lshft(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine lshft4
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*8 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'lshft_longlong'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * (bit_size(C(i)) + 1))
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = lshft(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (lshft(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine rshft1
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*1 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'rshft_char'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * (bit_size(C(i)) + 1))
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = rshft(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (rshft(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine rshft2
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*2 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'rshft_short'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * (bit_size(C(i)) + 1))
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = rshft(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (rshft(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine rshft3
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*4 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'rshft_long'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * (bit_size(C(i)) + 1))
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = rshft(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (rshft(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine rshft4
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*8 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'rshft_longlong'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * (bit_size(C(i)) + 1))
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = rshft(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (rshft(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ishftc9
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*1 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'bshftc_char'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * (bit_size(C(i)) + 1))
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = bshftc(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (bshftc(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ishftc10
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*1 A(N), B(N), C(N), D(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'bshftc_char'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          D(i) = int(tmp * bit_size(D(i)) + 1)
          call random_number(tmp)
          C(i) = int(tmp * D(i))
      enddo

!dvm$ actual(B, C, D)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = bshftc(B(i), C(i), D(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (bshftc(B(i), C(i), D(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ishftc11
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*2 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'iishftc_short'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * (bit_size(C(i)) + 1))
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = iishftc(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (iishftc(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ishftc12
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*2 A(N), B(N), C(N), D(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'iishftc_short'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          D(i) = int(tmp * bit_size(D(i)) + 1)
          call random_number(tmp)
          C(i) = int(tmp * D(i))
      enddo

!dvm$ actual(B, C, D)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = iishftc(B(i), C(i), D(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (iishftc(B(i), C(i), D(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ishftc13
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*2 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'hshftc_short'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * (bit_size(C(i)) + 1))
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = hshftc(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (hshftc(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ishftc14
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*2 A(N), B(N), C(N), D(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'hshftc_short'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          D(i) = int(tmp * bit_size(D(i)) + 1)
          call random_number(tmp)
          C(i) = int(tmp * D(i))
      enddo

!dvm$ actual(B, C, D)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = hshftc(B(i), C(i), D(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (hshftc(B(i), C(i), D(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif
      
      end

C -------------------------------------------------

      subroutine ishftc15
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*4 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'jishftc_long'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * (2 * bit_size(C(i)) + 1) - bit_size(C(i)))
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = jishft(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (jishft(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ishftc16
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*4 A(N), B(N), C(N), D(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'jishftc_long'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          D(i) = int(tmp * bit_size(D(i)) + 1)
          call random_number(tmp)
          C(i) = int(tmp * D(i))
      enddo

!dvm$ actual(B, C, D)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = jishftc(B(i), C(i), D(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (jishftc(B(i), C(i), D(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ishftc17
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*8 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'kishftc_longlong'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * (bit_size(C(i)) + 1))
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = kishftc(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (kishftc(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ishftc18
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*8 A(N), B(N), C(N), D(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'kishftc_longlong'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          D(i) = int(tmp * bit_size(D(i)) + 1)
          call random_number(tmp)
          C(i) = int(tmp * D(i))
      enddo

!dvm$ actual(B, C, D)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = kishftc(B(i), C(i), D(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (kishftc(B(i), C(i), D(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ishl1
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*1 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'ishl_char'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * (bit_size(C(i)) + 1))
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = ishl(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (ishl(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ishl2
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*2 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'ishl_short'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * (bit_size(C(i)) + 1))
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = ishl(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (ishl(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ishl3
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*4 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'ishl_long'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * (bit_size(C(i)) + 1))
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = ishl(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (ishl(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine ishl4
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*8 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'ishl_longlong'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * (bit_size(C(i)) + 1))
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = ishl(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (ishl(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine log103
      integer, parameter :: N = 256, ER = N + 1, W = 99, S = 1
      real*8, parameter :: EPS = 1e-6
      character*24 tname
      complex A(N), B(N)
      real x, y
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      complex tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'log10_complexf'
      erri = ER

      do i = 1, N
          call random_number(x)
          call random_number(y)
          B(i) = cmplx(x * W + S, y * W + S)
      enddo

!dvm$ actual(B)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = log10(B(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri)), private(tmp)
      do i = 1, N
          tmp = log10(B(i))
          if (abs(tmp - A(i))/abs(tmp) .gt. EPS) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine log104
      integer, parameter :: N = 256, ER = N + 1, W = 99, S = 1
      real*8, parameter :: EPS = 1d-15
      character*24 tname
      double complex A(N), B(N)
      real*8 x, y
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      double complex tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'log10_complexd'
      erri = ER

      do i = 1, N
          call random_number(x)
          call random_number(y)
          B(i) = dcmplx(x * W + S, y * W + S)
      enddo

!dvm$ actual(B)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = log10(B(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri)), private(tmp)
      do i = 1, N
          tmp = log10(B(i))
          if (abs(tmp - A(i))/abs(tmp) .gt. EPS) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine log107
      integer, parameter :: N = 256, ER = N + 1, W = 99, S = 1
      real, parameter :: EPS = 1e-6
      character*24 tname
      complex A(N), B(N)
      real x, y
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      complex tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'clog10_complexf'
      erri = ER

      do i = 1, N
          call random_number(x)
          call random_number(y)
          B(i) = cmplx(x * W + S, y * W + S)
      enddo

!dvm$ actual(B)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = clog10(B(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri)), private(tmp)
      do i = 1, N
          tmp = clog10(B(i))
          if (abs(tmp - A(i))/abs(tmp) .gt. EPS) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine log108
      integer, parameter :: N = 256, ER = N + 1, W = 99, S = 1
      real*8, parameter :: EPS = 1d-15
      character*24 tname
      double complex A(N), B(N)
      real*8 x, y
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      double complex tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'cdlog10_complexd'
      erri = ER

      do i = 1, N
          call random_number(x)
          call random_number(y)
          B(i) = dcmplx(x * W + S, y * W + S)
      enddo

!dvm$ actual(B)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = cdlog10(B(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri)), private(tmp)
      do i = 1, N
          tmp = cdlog10(B(i))
          if (abs(tmp - A(i))/abs(tmp) .gt. EPS) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine max10_
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*2 A(N), B(N), C(N), D(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'imax0_short'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * W + S)
          call random_number(tmp)
          D(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B, C, D)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = imax0(B(i), C(i), D(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (imax0(B(i), C(i), D(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine max11_
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*4 A(N), B(N), C(N), D(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'jmax0_long'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * W + S)
          call random_number(tmp)
          D(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B, C, D)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = jmax0(B(i), C(i), D(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (jmax0(B(i), C(i), D(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine max12_
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*8 A(N), B(N), C(N), D(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'kmax0_longlong'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * W + S)
          call random_number(tmp)
          D(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B, C, D)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = kmax0(B(i), C(i), D(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (kmax0(B(i), C(i), D(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine max14_
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*2 A(N)
      real B(N), C(N), D(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'imax1_float'
      erri = ER

      do i = 1, N
          call random_number(B(i))
          B(i) = B(i) * W + S
          call random_number(C(i))
          C(i) = C(i) * W + S
          call random_number(D(i))
          D(i) = D(i) * W + S
      enddo

!dvm$ actual(B, C, D)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = imax1(B(i), C(i), D(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (imax1(B(i), C(i), D(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine max15_
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*4 A(N)
      real B(N), C(N), D(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'jmax1_float'
      erri = ER

      do i = 1, N
          call random_number(B(i))
          B(i) = B(i) * W + S
          call random_number(C(i))
          C(i) = C(i) * W + S
          call random_number(D(i))
          D(i) = D(i) * W + S
      enddo

!dvm$ actual(B, C, D)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = jmax1(B(i), C(i), D(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (jmax1(B(i), C(i), D(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine max16_
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*8 A(N)
      real B(N), C(N), D(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'kmax1_float'
      erri = ER

      do i = 1, N
          call random_number(B(i))
          B(i) = B(i) * W + S
          call random_number(C(i))
          C(i) = C(i) * W + S
          call random_number(D(i))
          D(i) = D(i) * W + S
      enddo

!dvm$ actual(B, C, D)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = kmax1(B(i), C(i), D(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (kmax1(B(i), C(i), D(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end
C -------------------------------------------------

      subroutine max18_
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      real A(N)
      integer*2 B(N), C(N), D(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'aimax0_short'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * W + S)
          call random_number(tmp)
          D(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B, C, D)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = aimax0(B(i), C(i), D(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (aimax0(B(i), C(i), D(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine max19_
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      real A(N)
      integer*4 B(N), C(N), D(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'ajmax0_long'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * W + S)
          call random_number(tmp)
          D(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B, C, D)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = ajmax0(B(i), C(i), D(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (ajmax0(B(i), C(i), D(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine max20_
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      real A(N)
      integer*8 B(N), C(N), D(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'akmax0_longlong'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * W + S)
          call random_number(tmp)
          D(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B, C, D)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = akmax0(B(i), C(i), D(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (akmax0(B(i), C(i), D(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine min10_
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*2 A(N), B(N), C(N), D(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'imin0_short'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * W + S)
          call random_number(tmp)
          D(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B, C, D)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = imin0(B(i), C(i), D(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (imin0(B(i), C(i), D(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine min11_
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*4 A(N), B(N), C(N), D(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'jmin0_long'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * W + S)
          call random_number(tmp)
          D(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B, C, D)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = jmin0(B(i), C(i), D(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (jmin0(B(i), C(i), D(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine min12_
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*8 A(N), B(N), C(N), D(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'kmin0_longlong'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * W + S)
          call random_number(tmp)
          D(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B, C, D)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = kmin0(B(i), C(i), D(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (kmin0(B(i), C(i), D(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end
C -------------------------------------------------

      subroutine min14_
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*2 A(N)
      real B(N), C(N), D(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'imin1_float'
      erri = ER

      do i = 1, N
          call random_number(B(i))
          B(i) = B(i) * W + S
          call random_number(C(i))
          C(i) = C(i) * W + S
          call random_number(D(i))
          D(i) = D(i) * W + S
      enddo

!dvm$ actual(B, C, D)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = imin1(B(i), C(i), D(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (imin1(B(i), C(i), D(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine min15_
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*4 A(N)
      real B(N), C(N), D(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'jmin1_float'
      erri = ER

      do i = 1, N
          call random_number(B(i))
          B(i) = B(i) * W + S
          call random_number(C(i))
          C(i) = C(i) * W + S
          call random_number(D(i))
          D(i) = D(i) * W + S
      enddo

!dvm$ actual(B, C, D)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = jmin1(B(i), C(i), D(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (jmin1(B(i), C(i), D(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine min16_
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*8 A(N)
      real B(N), C(N), D(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'kmin1_float'
      erri = ER

      do i = 1, N
          call random_number(B(i))
          B(i) = B(i) * W + S
          call random_number(C(i))
          C(i) = C(i) * W + S
          call random_number(D(i))
          D(i) = D(i) * W + S
      enddo

!dvm$ actual(B, C, D)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = kmin1(B(i), C(i), D(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (kmin1(B(i), C(i), D(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine min18_
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      real A(N)
      integer*2 B(N), C(N), D(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'aimin0_short'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * W + S)
          call random_number(tmp)
          D(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B, C, D)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = aimin0(B(i), C(i), D(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (aimin0(B(i), C(i), D(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine min19_
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      real A(N)
      integer*4 B(N), C(N), D(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'ajmin0_long'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * W + S)
          call random_number(tmp)
          D(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B, C, D)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = ajmin0(B(i), C(i), D(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (ajmin0(B(i), C(i), D(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine min20_
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      real A(N)
      integer*8 B(N), C(N), D(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'akmin0_longlong'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * W + S)
          call random_number(tmp)
          D(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B, C, D)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = akmin0(B(i), C(i), D(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (akmin0(B(i), C(i), D(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine mod5
      integer, parameter :: N = 256, ER = N + 1, W1 = 200, S1 = -100, W2
     & = 99, S2 = 1
      character*24 tname
      integer*1 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'bmod_char'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W1 + S1)
          call random_number(tmp)
          C(i) = int(tmp * W2 + S2)
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = bmod(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (bmod(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine mod6
      integer, parameter :: N = 256, ER = N + 1, W1 = 200, S1 = -100, W2
     & = 99, S2 = 1
      character*24 tname
      integer*2 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'imod_short'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W1 + S1)
          call random_number(tmp)
          C(i) = int(tmp * W2 + S2)
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = imod(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (imod(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine mod7
      integer, parameter :: N = 256, ER = N + 1, W1 = 200, S1 = -100, W2
     & = 99, S2 = 1
      character*24 tname
      integer*2 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'hmod_short'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W1 + S1)
          call random_number(tmp)
          C(i) = int(tmp * W2 + S2)
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = hmod(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (hmod(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine mod8
      integer, parameter :: N = 256, ER = N + 1, W1 = 200, S1 = -100, W2
     & = 99, S2 = 1
      character*24 tname
      integer*4 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'jmod_long'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W1 + S1)
          call random_number(tmp)
          C(i) = int(tmp * W2 + S2)
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = jmod(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (jmod(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine mod9
      integer, parameter :: N = 256, ER = N + 1, W1 = 200, S1 = -100, W2
     & = 99, S2 = 1
      character*24 tname
      integer*8 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'kmod_longlong'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W1 + S1)
          call random_number(tmp)
          C(i) = int(tmp * W2 + S2)
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = kmod(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (kmod(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine not5
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*1 A(N), B(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'bnot_char'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = bnot(B(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (bnot(B(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine not6
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*2 A(N), B(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'inot_short'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = inot(B(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (inot(B(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine not7
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*2 A(N), B(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'hnot_short'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = hnot(B(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (hnot(B(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine not8
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*4 A(N), B(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'jnot_long'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = jnot(B(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (jnot(B(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine not9
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*8 A(N), B(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'knot_longlong'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = knot(B(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (knot(B(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine sign7
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*1 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'isign_char'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = isign(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (isign(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine sign8
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*2 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'isign_short'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = isign(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (isign(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine sign10
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*8 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'isign_longlong'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = isign(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (isign(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end
C -------------------------------------------------

      subroutine sign11
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*1 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'bsign_char'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = bsign(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (bsign(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine sign12
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*2 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'iisign_short'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = iisign(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (iisign(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine sign13
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*2 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'hsign_short'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = hsign(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (hsign(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine sign14
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*4 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'jisign_long'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = jisign(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (jisign(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine sign15
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      integer*8 A(N), B(N), C(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'kisign_longlong'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
          call random_number(tmp)
          C(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B, C)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = kisign(B(i), C(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (kisign(B(i), C(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine real9
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      real A(N)
      integer*2 B(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'floati_short'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = floati(B(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (floati(B(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine real11
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      real A(N)
      integer*4 B(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'floatj_long'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = floatj(B(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (floatj(B(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine real12
      integer, parameter :: N = 256, ER = N + 1, W = 200, S = -100
      character*24 tname
      real A(N)
      integer*8 B(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'floatk_longlong'
      erri = ER

      do i = 1, N
          call random_number(tmp)
          B(i) = int(tmp * W + S)
      enddo

!dvm$ actual(B)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = floatk(B(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri))
      do i = 1, N
          if (floatk(B(i)) .ne. A(i)) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end


C -------------------------------------------------

      subroutine tan6
      integer, parameter :: N = 256, ER = N + 1, W = 2, S = -1
      real, parameter :: EPS = 1e-6
      character*24 tname
      complex A(N), B(N)
      real x, y
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      complex tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'ctan_complexf'
      erri = ER

      do i = 1, N
          call random_number(x)
          call random_number(y)
          B(i) = cmplx(x * W + S, y * W + S)
      enddo

!dvm$ actual(B)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = ctan(B(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri)), private(tmp)
      do i = 1, N
          tmp = ctan(B(i))
          if (abs(tmp - A(i))/abs(tmp) .gt. EPS) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine tan7
      integer, parameter :: N = 256, ER = N + 1, W = 2, S = -1
      real*8, parameter :: EPS = 1d-15
      character*24 tname
      double complex A(N), B(N)
      real*8 x, y
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      double complex tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'cdtan_complexd'
      erri = ER

      do i = 1, N
          call random_number(x)
          call random_number(y)
          B(i) = dcmplx(x * W + S, y * W + S)
      enddo

!dvm$ actual(B)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = cdtan(B(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri)), private(tmp)
      do i = 1, N
          tmp = cdtan(B(i))
          if (abs(tmp - A(i))/abs(tmp) .gt. EPS) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine tan8
      integer, parameter :: N = 256, ER = N + 1, W = 2, S = -1
      real*8, parameter :: EPS = 1d-15
      character*24 tname
      double complex A(N), B(N)
      real*8 x, y
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      double complex tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'ztan_complexd'
      erri = ER

      do i = 1, N
          call random_number(x)
          call random_number(y)
          B(i) = dcmplx(x * W + S, y * W + S)
      enddo

!dvm$ actual(B)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = ztan(B(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri)), private(tmp)
      do i = 1, N
          tmp = ztan(B(i))
          if (abs(tmp - A(i))/abs(tmp) .gt. EPS) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end


C -------------------------------------------------

      subroutine sind1
      integer, parameter :: N = 256, ER = N + 1, W = 720, S = -360
      real, parameter :: EPS = 1e-6
      character*24 tname
      real A(N), B(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'sind_float'
      erri = ER

      do i = 1, N
          call random_number(B(i))
          B(i) = B(i) * W + S
      enddo

!dvm$ actual(B)	  
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = sind(B(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri)), private(tmp)
      do i = 1, N
          tmp = sind(B(i))
          if (abs(tmp - A(i))/abs(tmp) .gt. EPS) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine sind2
      integer, parameter :: N = 256, ER = N + 1, W = 720, S = -360
      real*8, parameter :: EPS = 1d-11
      character*24 tname
      real*8 A(N), B(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real*8 tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'sind_double'
      erri = ER

      do i = 1, N
          call random_number(B(i))
          B(i) = B(i) * W + S
      enddo

!dvm$ actual(B)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = sind(B(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri)), private(tmp)
      do i = 1, N
          tmp = sind(B(i))
          if (abs(tmp - A(i))/abs(tmp) .gt. EPS) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine sind3
      integer, parameter :: N = 256, ER = N + 1, W = 720, S = -360
      real*8, parameter :: EPS = 1d-11
      character*24 tname
      real*8 A(N), B(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real*8 tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'dsind_double'
      erri = ER

      do i = 1, N
          call random_number(B(i))
          B(i) = B(i) * W + S
      enddo

!dvm$ actual(B)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = dsind(B(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri)), private(tmp)
      do i = 1, N
          tmp = dsind(B(i))
          if (abs(tmp - A(i))/abs(tmp) .gt. EPS) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine tand1
      integer, parameter :: N = 256, ER = N + 1, W = 720, S = -360
      real, parameter :: EPS = 1e-6
      character*24 tname
      real A(N), B(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'tand_float'
      erri = ER

      do i = 1, N
          call random_number(B(i))
          B(i) = B(i) * W + S
      enddo

!dvm$ actual(B)	  
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = tand(B(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri)), private(tmp)
      do i = 1, N
          tmp = tand(B(i))
          if (abs(tmp - A(i))/abs(tmp) .gt. EPS) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine tand2
      integer, parameter :: N = 256, ER = N + 1, W = 720, S = -360
      real*8, parameter :: EPS = 1d-11
      character*24 tname
      real*8 A(N), B(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real*8 tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'tand_double'
      erri = ER

      do i = 1, N
          call random_number(B(i))
          B(i) = B(i) * W + S
      enddo

!dvm$ actual(B)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = tand(B(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri)), private(tmp)
      do i = 1, N
          tmp = tand(B(i))
          if (abs(tmp - A(i))/abs(tmp) .gt. EPS) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end

C -------------------------------------------------

      subroutine tand3
      integer, parameter :: N = 256, ER = N + 1, W = 720, S = -360
      real*8, parameter :: EPS = 1d-11
      character*24 tname
      real*8 A(N), B(N)
      integer erri, i, asize, clock
      integer, allocatable :: seed(:)
      real*8 tmp

!dvm$ distribute A(BLOCK)

      call random_seed(size = asize)
      allocate(seed(asize))
      call system_clock(count = clock)
      seed = clock + 37 * (/(i - 1, i = 1, asize)/)
      call random_seed(put = seed)
      deallocate(seed)

      tname = 'dtand_double'
      erri = ER

      do i = 1, N
          call random_number(B(i))
          B(i) = B(i) * W + S
      enddo

!dvm$ actual(B)
!dvm$ region
!dvm$ parallel (i) on A(i)
      do i = 1, N
          A(i) = dtand(B(i))
      enddo
!dvm$ end region
!dvm$ get_actual(A)

!dvm$ parallel (i) on A(i), reduction(min(erri)), private(tmp)
      do i = 1, N
          tmp = dtand(B(i))
          if (abs(tmp - A(i))/abs(tmp) .gt. EPS) then
              erri = min(erri, i)
          endif
      enddo

      if (erri .eq. ER) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      
      end


C -------------------------------------------------

      subroutine ansyes(name)
      character*24 name
      print *, name, '  -  complete'
      end

      subroutine ansno(name)
      character*24 name
      print *, name, '  -  ***error'
      end
