
#ifndef AT_VIRTUAL
#define AT_VIRTUAL(f)                    virtual f
#endif

#ifndef AT_VIRTUAL_OVERRIDE
#define AT_VIRTUAL_OVERRIDE(f)            virtual f override
#endif

#ifndef AT_VIRTUAL_OVERRIDE_FINAL
#define AT_VIRTUAL_OVERRIDE_FINAL(f)    virtual f override final
#endif

#ifndef AT_PURE_VIRTUAL
#define AT_PURE_VIRTUAL(f)                virtual f = 0
#endif

#ifndef AT_INHERIT
#define AT_INHERIT(c)    : public c
#endif