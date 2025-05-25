      program copy21
      parameter (ERR=10000, L=10)
      integer A(L,L),X(L,L), C(L,L),B(L,L),D(L,L)
      integer:: ERRI=ERR
!DVM$ DISTRIBUTE (BLOCK,BLOCK) :: A
!DVM$ ALIGN C(I,J) WITH A(I,J)
!DVM$ ALIGN D(I,J) WITH A(I,J)

      character*6:: tname ='copy21'
      print *,'========  START OF copy21  =================='
      
      
      do J=1,L
      do I=1,L
        X(I,J)=I+J
      enddo
      enddo 
      B = 0
      C(:,:)=B(:,:)
      D(1:L,1) = X(1:L,1)
      C(1:L,1) = D(1:L,1)


!DVM$ PARALLEL (J,I) ON C(I,J), REDUCTION (min(ERRI))
      do J=1,L
      do I=1,L
        if (J.eq.1 .and. X(I,1) .ne. C(I,1)) then     
          ERRI = min (I,ERRI)
        else  if(J.ne.1 .and. C(I,J) .ne. 0)  then
          ERRI = min (I,ERRI)
        endif        
      enddo
      enddo
      if (ERRI .eq.ERR) then     
          call ansyes(tname)
      else
          call ansno(tname)
      endif 
      print *,'======== END OF copy21 ======================'
      end

      subroutine ansyes(name)
      character*6 name
      print *,name,'  -  complete'
      end
      subroutine ansno(name)
      character*6 name
      print *,name,'  -  ***error'
      end
