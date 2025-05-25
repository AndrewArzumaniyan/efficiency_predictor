	PROGRAM    taskst12
!	rectangular grid is distributed on two blocks
!
!	
	INTEGER,PARAMETER :: K=8, N1=4, ITMAX=20, N2=K-N1, ER=10000
	REAL,ALLOCATABLE :: A(:,:),A1(:,:),A2(:,:)
        REAL,ALLOCATABLE :: B(:,:),B1(:,:),B2(:,:) 
        INTEGER,DIMENSION(2) :: LP,HP
        INTEGER :: ERRT
        CHARACTER*8:: TNAME='taskst12'
CDVM$	PROCESSORS    P(NUMBER_OF_PROCESSORS( ))
CDVM$	TASK  MB( 2 )
CDVM$	DISTRIBUTE   A(*,BLOCK)
CDVM$	ALIGN   B( I, J )  WITH  A( I, J )
CDVM$	DISTRIBUTE  ::  A1, A2
CDVM$	ALIGN ::  B1,B2
   
        PRINT *,  '===START OF taskst12 ====================='
        CALL DPT(LP,HP,2)
CDVM$	MAP  MB( 1 )  ONTO  P( LP(1) : HP(1) )
        ALLOCATE(A1(N1+1,K))
CDVM$	REDISTRIBUTE  A1( *, BLOCK )  ONTO  MB( 1 )
        ALLOCATE(B1(N1+1,K))
CDVM$	REALIGN   B1( I, J )  WITH  A1( I, J ) 

CDVM$	MAP  MB( 2 )   ONTO  P( LP(2) : HP(2) )
        ALLOCATE(A2(N2+1,K))
CDVM$	REDISTRIBUTE  A2( *, BLOCK )  ONTO  MB( 2 )
        ALLOCATE(B2(N2+1,K))
CDVM$	REALIGN   B2( I, J )  WITH  A2( I, J )

        ALLOCATE(A(K,K),B(K,K))

!		Initialization
!DVM$   TASK_REGION MB
!DVM$   ON MB(1)
!DVM$   REGION
!DVM$	PARALLEL    ( J, I )   ON   A1(I, J)
	DO  J  =  1, K
	   DO I  =  1, N1
		IF(I.EQ.1 .OR. J.EQ.1 .OR. J.EQ.K) THEN
			A1(I, J) = 0.
			B1(I, J) = 0.
		ELSE
			B1(I, J)  = 1. + I + J 
			A1(I, J) = B1(I, J)
		ENDIF
           ENDDO
        ENDDO
!DVM$   END REGION
!DVM$   END ON
!DVM$   ON MB(2)
!DVM$   REGION
!DVM$	PARALLEL    ( J, I )   ON   A2(I, J)
	DO  J  =  1, K
           DO  I  =  2, N2+1
		IF(I.EQ.N2+1 .OR. J.EQ.1 .OR. J.EQ.K) THEN
			A2(I, J) = 0.
			B2(I, J) = 0.
		ELSE
			B2(I, J)  = 1. + (I+N1-1) + J 
			A2(I, J) = B2(I, J)
		ENDIF
           ENDDO
        ENDDO
!DVM$   END REGION
!DVM$   END ON
!DVM$   END TASK_REGION
      
	DO  2   IT  =  1, ITMAX

!		exchange bounds
!DVM$   GET_ACTUAL (B2(2,:),B1(N1, :))
        A1(N1+1,:) = B2(2, :)
        A2(1, :)   = B1(N1, :)
!DVM$   ACTUAL (A2(1, :),A1(N1+1,:))
!DVM$	TASK_REGION MB
!DVM$	ON   MB( 1 )
!DVM$   REGION
!DVM$	PARALLEL    ( J, I )   ON   B1(I, J),
!DVM$*  SHADOW_RENEW ( A1 )
	DO  J  =  2, K-1
            DO  I  =  2, N1
                 B1(I, J)=(A1(I-1, J) + A1(I,J-1) + 
     *           A1(I+1,J) + A1(I,J+1))/4
            ENDDO
        ENDDO

!DVM$	PARALLEL    ( J, I )   ON   A1(I, J)
	DO  J  =  2, K-1
   	    DO  I  =  2, N1
                A1(I, J) =  B1( I, J )
            ENDDO
        ENDDO 
!DVM$   END REGION
!DVM$	END ON
!DVM$	ON   MB( 2 )
!DVM$   REGION
!DVM$	PARALLEL    ( J, I )   ON   B2(I, J),
!DVM$*  SHADOW_RENEW ( A2 )
	DO  J  =  2, K-1
            DO  I  =  2, N2
                B2(I,J) = (A2(I-1,J) + A2(I,J-1) + 
     *          A2(I+1,J) + A2(I,J+1))/4
            ENDDO
        ENDDO          
!DVM$	PARALLEL    ( J, I )   ON   A2(I, J)
	DO  J  =  2, K-1
            DO  I  =  2, N2
                A2(I, J) =  B2( I, J )
            ENDDO
        ENDDO 
!DVM$   END REGION
!DVM$	END ON
!DVM$	END  TASK_REGION
2       CONTINUE
!1-task JACOBI   

!DVM$   REGION   
!DVM$   PARALLEL    (J,I)   ON   A(I, J)
!		nest of two parallel loops, iteration (i,j) will be executed on 
!		processor, which is owner of element A(i,j) 
            DO  J  =  1, K
                DO  I  =  1, K
                    A(I,  J)  =  0.
                    IF(I.EQ.1.OR.J.EQ.1.OR.I.EQ.K.OR.J.EQ.K) THEN
                       B(I,  J) = 0.
                    ELSE
                       B(I,  J)  = ( 1. + I + J )
                    ENDIF
                ENDDO
            ENDDO
!DVM$   END REGION            
        DO  IT  =  1,  ITMAX
!DVM$  REGION       
!DVM$   PARALLEL  (J,  I)   ON  A(I,  J)
!		variable EPS is used for calculation of maximum value
                  DO  J  =  2, K-1
                      DO  I  =  2, K-1
                         A(I, J)  =  B(I, J)
                      ENDDO
                  ENDDO
!DVM$   PARALLEL  (J,  I)   ON  B(I,  J),   SHADOW_RENEW   (A)
!		Copying shadow elements of array A from 
!		neighbouring processors before loop execution
                  DO  J = 2,  K-1
                      DO  I = 2,  K-1
        B(I, J) =  (A( I-1, J ) + A( I, J-1 ) + A( I+1, J)+
     *                        A( I, J+1 )) / 4
                      ENDDO
                  ENDDO
!DVM$ END REGION
        ENDDO
!DVM$ GET_ACTUAL (B,B1,B2)
!   compare 2-task JACOBI with 1-task JACOBI
      A(2:N1,:) = B1(2:N1,:)
      A(N1+1:N1+N2-1,:) = B2(2:N2,:)
      ERRT = ER
!DVM$ PARALLEL (I,J)  ON B(I,J), REDUCTION(MIN(ERRT))
      DO I = 2, K-1
         DO J = 2, K-1
            IF(A(I,J) .NE. B(I,J)) THEN
               ERRT = MIN(ERRT,I)
            ENDIF 
         ENDDO
      ENDDO 
      IF (ERRT .EQ. ER) THEN     
          CALL ANSYES(TNAME)
      ELSE
          CALL ANSNO(TNAME)
      ENDIF 

      DEALLOCATE (B,B1,B2,A,A1,A2)
      PRINT *,  '=== END OF taskst12 ====================='

      END

      SUBROUTINE DPT(LP,HP,NT)
!     distributing processors for NT tasks (NT = 2)      
      INTEGER LP(2), HP(2)
      NUMBER_OF_PROCESSORS() = 1
!DVM$ DEBUG 1 (D = 0)
      NP = NUMBER_OF_PROCESSORS()
      NTP = NP/NT
      IF(NP.EQ.1) THEN
         LP(1) = 1
         HP(1) = 1
         LP(2) = 1
         HP(2) = 1
      ELSE
         LP(1) = 1
         HP(1) = NTP
         LP(2) = NTP+1
         HP(2) = NP
      END IF
!DVM$ ENDDEBUG 1
      END
C -------------------------------------------------

      SUBROUTINE ANSYES(NAME)
      CHARACTER*8 NAME
      PRINT *, NAME, '  -  complete'
      END
      SUBROUTINE ANSNO (NAME)
      CHARACTER*8 NAME
      PRINT *, NAME, '  -  ***error'
      END