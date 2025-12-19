# Global variables
#
# This file declares variables that are used in package functions
# but are not explicitly defined, to avoid R CMD check warnings.

if (getRversion() >= "2.15.1") {
  utils::globalVariables(c("self"))
}
