
SET(MY_LIBRARY_SRCS  
MathFinanceWorkshop.cpp 
)

ADD_LIBRARY(MathFinanceWorkshopLIB SHARED
${MY_LIBRARY_SRCS} 
)

TARGET_LINK_LIBRARIES( MathFinanceWorkshopLIB )

ADD_LIBRARY(MathFinanceWorkshopLIB_static STATIC
${MY_LIBRARY_SRCS} 
)

TARGET_LINK_LIBRARIES( MathFinanceWorkshopLIB_static )

