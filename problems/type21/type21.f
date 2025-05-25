      program type21
      integer(4),parameter:: n=4,m=4,l=4,err=100
      real ,dimension (n,m,l):: a     
      integer::k,p,q,err1,err2,err3
      type OBJECT
         character (15) name
         integer st(n)
         integer bl(n,m)
         integer matr(n,m)
      end type OBJECT

      integer, dimension(n,m) :: a1, b, c
      integer, dimension(n) :: qq

CDVM$ distribute (BLOCK, *):: a1 
CDVM$ align qq(i) with a1( i, *)
CDVM$ align (i,j) with a1(i,j):: c,b
      type(OBJECT) :: GR,OTD
CDVM$ ASYNCID Y

      print *,'====== START OF TYPE21 =========='
! Testing of different variants of deffinitions a
      do k=1,n
        do p=1,m
          do q=1,l
             a(k,p,q)=10+k+p+q
          end do
        end do
      end do
      a=1; a(1:n,1:m,1:l)=10+n+m+l!!; print*,a
      a=1; forall(k=1:n,p=1:m,q=1:l) a(k,p,q) = 10+k+p+q;
      do k=1,n
        do p=1,m
          a1(k,p)=k
        end do
      end do
      c=0

CDVM$ ASYNCHRONOUS Y
      qq(:)= a1(:,2)
      b(:,:) = a1(:,:)
      c(1:2,:) = a1(3:4,:)
CDVM$ END ASYNCHRONOUS
      GR%st=(/1,2,3,4/)

      do k=1,n
        do p=1,m
          GR%matr(k,p)=k
        end do
      end do

      GR%bl=reshape((/3,4,0,0,3,4,0,0,3,4,0,0,3,4,0,0/),(/4,4/))

CDVM$ ASYNCWAIT Y
      err1=err;err2=err;err3=err
CDVM$ parallel(i) on qq(i),reduction (min(err1))
      do i=1,n
        if (qq(i) .ne. GR%st(i)) then        
          err1 = min (i,err1)
        endif 
      enddo

CDVM$ parallel(i,j) on a1(i,j), reduction (min(err2))
      do i=1,n
        do j=1,m
          if (a1(i,j) .ne. GR%matr(i,j)) then     
            err2 = min (i,err2)
          endif
        enddo
      enddo

CDVM$ parallel(i,j) on c(i,j), reduction (min(err3))
      do i=1,n
        do j=1,m
           if (c(i,j) .ne.GR%bl(i,j)) then     
             err3 = min (i,err3)
           endif
        enddo
      enddo
  
      if ((err1 .ne. err).OR.(err2 .ne.err).OR.(err3.ne.err)) then
      print *,'type21 - ***error '
      else
      print *,'type21 - complete'
      endif
      print *,'=== END OF TYPE21 ======================='

      end

