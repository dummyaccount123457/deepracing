/* deepf1H.h
   Generated by gSOAP 2.8.61 for .\cpu_times.h

gSOAP XML Web services tools
Copyright (C) 2000-2018, Robert van Engelen, Genivia Inc. All Rights Reserved.
The soapcpp2 tool and its generated software are released under the GPL.
This program is released under the GPL with the additional exemption that
compiling, linking, and/or using OpenSSL is allowed.
--------------------------------------------------------------------------------
A commercial use license is available from Genivia Inc., contact@genivia.com
--------------------------------------------------------------------------------
*/

#ifndef deepf1H_H
#define deepf1H_H
#include "deepf1_gsoap/deepf1_gsoap_stub.h"

namespace deepf1 {
#ifndef WITH_NOIDREF
SOAP_FMAC3 void SOAP_FMAC4 soap_markelement(struct soap*, const void*, int);
SOAP_FMAC3 int SOAP_FMAC4 soap_putindependent(struct soap*);
SOAP_FMAC3 int SOAP_FMAC4 soap_getindependent(struct soap*);
#endif
SOAP_FMAC3 void * SOAP_FMAC4 soap_getelement(struct soap*, int*);
SOAP_FMAC3 int SOAP_FMAC4 soap_putelement(struct soap*, const void*, const char*, int, int);
SOAP_FMAC3 void * SOAP_FMAC4 soap_dupelement(struct soap*, const void*, int);
SOAP_FMAC3 void SOAP_FMAC4 soap_delelement(const void*, int);
SOAP_FMAC3 int SOAP_FMAC4 soap_ignore_element(struct soap*);

SOAP_FMAC3 const char ** SOAP_FMAC4 soap_faultcode(struct soap *soap);
SOAP_FMAC3 void * SOAP_FMAC4 deepf1_instantiate(struct soap*, int, const char*, const char*, size_t*);
SOAP_FMAC3 int SOAP_FMAC4 deepf1_fdelete(struct soap *soap, struct soap_clist*);
SOAP_FMAC3 int SOAP_FMAC4 deepf1_fbase(int, int);
SOAP_FMAC3 void SOAP_FMAC4 deepf1_finsert(struct soap*, int, int, void*, size_t, const void*, void**);

#ifndef SOAP_TYPE_deepf1_byte_DEFINED
#define SOAP_TYPE_deepf1_byte_DEFINED

inline void soap_default_byte(struct soap *soap, char *a)
{
	(void)soap; /* appease -Wall -Werror */
#ifdef SOAP_DEFAULT_byte
	*a = SOAP_DEFAULT_byte;
#else
	*a = (char)0;
#endif
}
SOAP_FMAC3 int SOAP_FMAC4 soap_out_byte(struct soap*, const char*, int, const char *, const char*);
SOAP_FMAC3 char * SOAP_FMAC4 soap_in_byte(struct soap*, const char*, char *, const char*);

SOAP_FMAC3 char * SOAP_FMAC4 soap_new_byte(struct soap *soap, int n = -1);
SOAP_FMAC3 int SOAP_FMAC4 soap_put_byte(struct soap*, const char *, const char*, const char*);

inline int soap_write_byte(struct soap *soap, char const*p)
{
	soap_free_temp(soap);
	if (p)
	{	if (soap_begin_send(soap) || deepf1::soap_put_byte(soap, p, "byte", "") || soap_end_send(soap))
			return soap->error;
	}
	return SOAP_OK;
}

inline int soap_PUT_byte(struct soap *soap, const char *URL, char const*p)
{
	soap_free_temp(soap);
	if (soap_PUT(soap, URL, NULL, "text/xml; charset=utf-8") || deepf1::soap_put_byte(soap, p, "byte", "") || soap_end_send(soap) || soap_recv_empty_response(soap))
		return soap_closesock(soap);
	return SOAP_OK;
}

inline int soap_POST_send_byte(struct soap *soap, const char *URL, char const*p)
{
	soap_free_temp(soap);
	if (soap_POST(soap, URL, NULL, "text/xml; charset=utf-8") || deepf1::soap_put_byte(soap, p, "byte", "") || soap_end_send(soap))
		return soap_closesock(soap);
	return SOAP_OK;
}
SOAP_FMAC3 char * SOAP_FMAC4 soap_get_byte(struct soap*, char *, const char*, const char*);

inline int soap_read_byte(struct soap *soap, char *p)
{
	if (p)
	{	if (soap_begin_recv(soap) || deepf1::soap_get_byte(soap, p, NULL, NULL) == NULL || soap_end_recv(soap))
			return soap->error;
	}
	return SOAP_OK;
}

inline int soap_GET_byte(struct soap *soap, const char *URL, char *p)
{
	if (soap_GET(soap, URL, NULL) || deepf1::soap_read_byte(soap, p) || soap_closesock(soap))
		return soap_closesock(soap);
	return SOAP_OK;
}

inline int soap_POST_recv_byte(struct soap *soap, char *p)
{
	if (deepf1::soap_read_byte(soap, p) || soap_closesock(soap))
		return soap_closesock(soap);
	return SOAP_OK;
}
#endif

#ifndef SOAP_TYPE_deepf1_int_DEFINED
#define SOAP_TYPE_deepf1_int_DEFINED

inline void soap_default_int(struct soap *soap, int *a)
{
	(void)soap; /* appease -Wall -Werror */
#ifdef SOAP_DEFAULT_int
	*a = SOAP_DEFAULT_int;
#else
	*a = (int)0;
#endif
}
SOAP_FMAC3 int SOAP_FMAC4 soap_out_int(struct soap*, const char*, int, const int *, const char*);
SOAP_FMAC3 int * SOAP_FMAC4 soap_in_int(struct soap*, const char*, int *, const char*);

SOAP_FMAC3 int * SOAP_FMAC4 soap_new_int(struct soap *soap, int n = -1);
SOAP_FMAC3 int SOAP_FMAC4 soap_put_int(struct soap*, const int *, const char*, const char*);

inline int soap_write_int(struct soap *soap, int const*p)
{
	soap_free_temp(soap);
	if (p)
	{	if (soap_begin_send(soap) || deepf1::soap_put_int(soap, p, "int", "") || soap_end_send(soap))
			return soap->error;
	}
	return SOAP_OK;
}

inline int soap_PUT_int(struct soap *soap, const char *URL, int const*p)
{
	soap_free_temp(soap);
	if (soap_PUT(soap, URL, NULL, "text/xml; charset=utf-8") || deepf1::soap_put_int(soap, p, "int", "") || soap_end_send(soap) || soap_recv_empty_response(soap))
		return soap_closesock(soap);
	return SOAP_OK;
}

inline int soap_POST_send_int(struct soap *soap, const char *URL, int const*p)
{
	soap_free_temp(soap);
	if (soap_POST(soap, URL, NULL, "text/xml; charset=utf-8") || deepf1::soap_put_int(soap, p, "int", "") || soap_end_send(soap))
		return soap_closesock(soap);
	return SOAP_OK;
}
SOAP_FMAC3 int * SOAP_FMAC4 soap_get_int(struct soap*, int *, const char*, const char*);

inline int soap_read_int(struct soap *soap, int *p)
{
	if (p)
	{	if (soap_begin_recv(soap) || deepf1::soap_get_int(soap, p, NULL, NULL) == NULL || soap_end_recv(soap))
			return soap->error;
	}
	return SOAP_OK;
}

inline int soap_GET_int(struct soap *soap, const char *URL, int *p)
{
	if (soap_GET(soap, URL, NULL) || deepf1::soap_read_int(soap, p) || soap_closesock(soap))
		return soap_closesock(soap);
	return SOAP_OK;
}

inline int soap_POST_recv_int(struct soap *soap, int *p)
{
	if (deepf1::soap_read_int(soap, p) || soap_closesock(soap))
		return soap_closesock(soap);
	return SOAP_OK;
}
#endif

#ifndef SOAP_TYPE_deepf1_long_DEFINED
#define SOAP_TYPE_deepf1_long_DEFINED

inline void soap_default_long(struct soap *soap, long *a)
{
	(void)soap; /* appease -Wall -Werror */
#ifdef SOAP_DEFAULT_long
	*a = SOAP_DEFAULT_long;
#else
	*a = (long)0;
#endif
}
SOAP_FMAC3 int SOAP_FMAC4 soap_out_long(struct soap*, const char*, int, const long *, const char*);
SOAP_FMAC3 long * SOAP_FMAC4 soap_in_long(struct soap*, const char*, long *, const char*);

SOAP_FMAC3 long * SOAP_FMAC4 soap_new_long(struct soap *soap, int n = -1);
SOAP_FMAC3 int SOAP_FMAC4 soap_put_long(struct soap*, const long *, const char*, const char*);

inline int soap_write_long(struct soap *soap, long const*p)
{
	soap_free_temp(soap);
	if (p)
	{	if (soap_begin_send(soap) || deepf1::soap_put_long(soap, p, "long", "") || soap_end_send(soap))
			return soap->error;
	}
	return SOAP_OK;
}

inline int soap_PUT_long(struct soap *soap, const char *URL, long const*p)
{
	soap_free_temp(soap);
	if (soap_PUT(soap, URL, NULL, "text/xml; charset=utf-8") || deepf1::soap_put_long(soap, p, "long", "") || soap_end_send(soap) || soap_recv_empty_response(soap))
		return soap_closesock(soap);
	return SOAP_OK;
}

inline int soap_POST_send_long(struct soap *soap, const char *URL, long const*p)
{
	soap_free_temp(soap);
	if (soap_POST(soap, URL, NULL, "text/xml; charset=utf-8") || deepf1::soap_put_long(soap, p, "long", "") || soap_end_send(soap))
		return soap_closesock(soap);
	return SOAP_OK;
}
SOAP_FMAC3 long * SOAP_FMAC4 soap_get_long(struct soap*, long *, const char*, const char*);

inline int soap_read_long(struct soap *soap, long *p)
{
	if (p)
	{	if (soap_begin_recv(soap) || deepf1::soap_get_long(soap, p, NULL, NULL) == NULL || soap_end_recv(soap))
			return soap->error;
	}
	return SOAP_OK;
}

inline int soap_GET_long(struct soap *soap, const char *URL, long *p)
{
	if (soap_GET(soap, URL, NULL) || deepf1::soap_read_long(soap, p) || soap_closesock(soap))
		return soap_closesock(soap);
	return SOAP_OK;
}

inline int soap_POST_recv_long(struct soap *soap, long *p)
{
	if (deepf1::soap_read_long(soap, p) || soap_closesock(soap))
		return soap_closesock(soap);
	return SOAP_OK;
}
#endif

#ifndef WITH_NOGLOBAL

#ifndef SOAP_TYPE_deepf1_SOAP_ENV__Fault_DEFINED
#define SOAP_TYPE_deepf1_SOAP_ENV__Fault_DEFINED
SOAP_FMAC3 void SOAP_FMAC4 soap_default_SOAP_ENV__Fault(struct soap*, struct SOAP_ENV__Fault *);
SOAP_FMAC3 void SOAP_FMAC4 soap_serialize_SOAP_ENV__Fault(struct soap*, const struct SOAP_ENV__Fault *);
SOAP_FMAC3 int SOAP_FMAC4 soap_out_SOAP_ENV__Fault(struct soap*, const char*, int, const struct SOAP_ENV__Fault *, const char*);
SOAP_FMAC3 struct SOAP_ENV__Fault * SOAP_FMAC4 soap_in_SOAP_ENV__Fault(struct soap*, const char*, struct SOAP_ENV__Fault *, const char*);
SOAP_FMAC1 struct SOAP_ENV__Fault * SOAP_FMAC2 soap_instantiate_SOAP_ENV__Fault(struct soap*, int, const char*, const char*, size_t*);

inline struct SOAP_ENV__Fault * soap_new_SOAP_ENV__Fault(struct soap *soap, int n = -1)
{
	return soap_instantiate_SOAP_ENV__Fault(soap, n, NULL, NULL, NULL);
}

inline struct SOAP_ENV__Fault * soap_new_req_SOAP_ENV__Fault(
	struct soap *soap)
{
	struct SOAP_ENV__Fault *_p = deepf1::soap_new_SOAP_ENV__Fault(soap);
	if (_p)
	{	deepf1::soap_default_SOAP_ENV__Fault(soap, _p);
	}
	return _p;
}

inline struct SOAP_ENV__Fault * soap_new_set_SOAP_ENV__Fault(
	struct soap *soap,
	char *faultcode,
	char *faultstring,
	char *faultactor,
	struct SOAP_ENV__Detail *detail,
	struct SOAP_ENV__Code *SOAP_ENV__Code,
	struct SOAP_ENV__Reason *SOAP_ENV__Reason,
	char *SOAP_ENV__Node,
	char *SOAP_ENV__Role,
	struct SOAP_ENV__Detail *SOAP_ENV__Detail)
{
	struct SOAP_ENV__Fault *_p = deepf1::soap_new_SOAP_ENV__Fault(soap);
	if (_p)
	{	deepf1::soap_default_SOAP_ENV__Fault(soap, _p);
		_p->faultcode = faultcode;
		_p->faultstring = faultstring;
		_p->faultactor = faultactor;
		_p->detail = detail;
		_p->SOAP_ENV__Code = SOAP_ENV__Code;
		_p->SOAP_ENV__Reason = SOAP_ENV__Reason;
		_p->SOAP_ENV__Node = SOAP_ENV__Node;
		_p->SOAP_ENV__Role = SOAP_ENV__Role;
		_p->SOAP_ENV__Detail = SOAP_ENV__Detail;
	}
	return _p;
}
SOAP_FMAC3 int SOAP_FMAC4 soap_put_SOAP_ENV__Fault(struct soap*, const struct SOAP_ENV__Fault *, const char*, const char*);

inline int soap_write_SOAP_ENV__Fault(struct soap *soap, struct SOAP_ENV__Fault const*p)
{
	soap_free_temp(soap);
	if (soap_begin_send(soap) || (deepf1::soap_serialize_SOAP_ENV__Fault(soap, p), 0) || deepf1::soap_put_SOAP_ENV__Fault(soap, p, "SOAP-ENV:Fault", "") || soap_end_send(soap))
			return soap->error;
	return SOAP_OK;
}

inline int soap_PUT_SOAP_ENV__Fault(struct soap *soap, const char *URL, struct SOAP_ENV__Fault const*p)
{
	soap_free_temp(soap);
	if (soap_PUT(soap, URL, NULL, "text/xml; charset=utf-8") || (deepf1::soap_serialize_SOAP_ENV__Fault(soap, p), 0) || deepf1::soap_put_SOAP_ENV__Fault(soap, p, "SOAP-ENV:Fault", "") || soap_end_send(soap) || soap_recv_empty_response(soap))
		return soap_closesock(soap);
	return SOAP_OK;
}

inline int soap_POST_send_SOAP_ENV__Fault(struct soap *soap, const char *URL, struct SOAP_ENV__Fault const*p)
{
	soap_free_temp(soap);
	if (soap_POST(soap, URL, NULL, "text/xml; charset=utf-8") || (deepf1::soap_serialize_SOAP_ENV__Fault(soap, p), 0) || deepf1::soap_put_SOAP_ENV__Fault(soap, p, "SOAP-ENV:Fault", "") || soap_end_send(soap))
		return soap_closesock(soap);
	return SOAP_OK;
}
SOAP_FMAC3 struct SOAP_ENV__Fault * SOAP_FMAC4 soap_get_SOAP_ENV__Fault(struct soap*, struct SOAP_ENV__Fault *, const char*, const char*);

inline int soap_read_SOAP_ENV__Fault(struct soap *soap, struct SOAP_ENV__Fault *p)
{
	if (p)
	{	deepf1::soap_default_SOAP_ENV__Fault(soap, p);
		if (soap_begin_recv(soap) || deepf1::soap_get_SOAP_ENV__Fault(soap, p, NULL, NULL) == NULL || soap_end_recv(soap))
			return soap->error;
	}
	return SOAP_OK;
}

inline int soap_GET_SOAP_ENV__Fault(struct soap *soap, const char *URL, struct SOAP_ENV__Fault *p)
{
	if (soap_GET(soap, URL, NULL) || deepf1::soap_read_SOAP_ENV__Fault(soap, p) || soap_closesock(soap))
		return soap_closesock(soap);
	return SOAP_OK;
}

inline int soap_POST_recv_SOAP_ENV__Fault(struct soap *soap, struct SOAP_ENV__Fault *p)
{
	if (deepf1::soap_read_SOAP_ENV__Fault(soap, p) || soap_closesock(soap))
		return soap_closesock(soap);
	return SOAP_OK;
}
#endif

#endif

#ifndef WITH_NOGLOBAL

#ifndef SOAP_TYPE_deepf1_SOAP_ENV__Reason_DEFINED
#define SOAP_TYPE_deepf1_SOAP_ENV__Reason_DEFINED
SOAP_FMAC3 void SOAP_FMAC4 soap_default_SOAP_ENV__Reason(struct soap*, struct SOAP_ENV__Reason *);
SOAP_FMAC3 void SOAP_FMAC4 soap_serialize_SOAP_ENV__Reason(struct soap*, const struct SOAP_ENV__Reason *);
SOAP_FMAC3 int SOAP_FMAC4 soap_out_SOAP_ENV__Reason(struct soap*, const char*, int, const struct SOAP_ENV__Reason *, const char*);
SOAP_FMAC3 struct SOAP_ENV__Reason * SOAP_FMAC4 soap_in_SOAP_ENV__Reason(struct soap*, const char*, struct SOAP_ENV__Reason *, const char*);
SOAP_FMAC1 struct SOAP_ENV__Reason * SOAP_FMAC2 soap_instantiate_SOAP_ENV__Reason(struct soap*, int, const char*, const char*, size_t*);

inline struct SOAP_ENV__Reason * soap_new_SOAP_ENV__Reason(struct soap *soap, int n = -1)
{
	return soap_instantiate_SOAP_ENV__Reason(soap, n, NULL, NULL, NULL);
}

inline struct SOAP_ENV__Reason * soap_new_req_SOAP_ENV__Reason(
	struct soap *soap)
{
	struct SOAP_ENV__Reason *_p = deepf1::soap_new_SOAP_ENV__Reason(soap);
	if (_p)
	{	deepf1::soap_default_SOAP_ENV__Reason(soap, _p);
	}
	return _p;
}

inline struct SOAP_ENV__Reason * soap_new_set_SOAP_ENV__Reason(
	struct soap *soap,
	char *SOAP_ENV__Text)
{
	struct SOAP_ENV__Reason *_p = deepf1::soap_new_SOAP_ENV__Reason(soap);
	if (_p)
	{	deepf1::soap_default_SOAP_ENV__Reason(soap, _p);
		_p->SOAP_ENV__Text = SOAP_ENV__Text;
	}
	return _p;
}
SOAP_FMAC3 int SOAP_FMAC4 soap_put_SOAP_ENV__Reason(struct soap*, const struct SOAP_ENV__Reason *, const char*, const char*);

inline int soap_write_SOAP_ENV__Reason(struct soap *soap, struct SOAP_ENV__Reason const*p)
{
	soap_free_temp(soap);
	if (soap_begin_send(soap) || (deepf1::soap_serialize_SOAP_ENV__Reason(soap, p), 0) || deepf1::soap_put_SOAP_ENV__Reason(soap, p, "SOAP-ENV:Reason", "") || soap_end_send(soap))
			return soap->error;
	return SOAP_OK;
}

inline int soap_PUT_SOAP_ENV__Reason(struct soap *soap, const char *URL, struct SOAP_ENV__Reason const*p)
{
	soap_free_temp(soap);
	if (soap_PUT(soap, URL, NULL, "text/xml; charset=utf-8") || (deepf1::soap_serialize_SOAP_ENV__Reason(soap, p), 0) || deepf1::soap_put_SOAP_ENV__Reason(soap, p, "SOAP-ENV:Reason", "") || soap_end_send(soap) || soap_recv_empty_response(soap))
		return soap_closesock(soap);
	return SOAP_OK;
}

inline int soap_POST_send_SOAP_ENV__Reason(struct soap *soap, const char *URL, struct SOAP_ENV__Reason const*p)
{
	soap_free_temp(soap);
	if (soap_POST(soap, URL, NULL, "text/xml; charset=utf-8") || (deepf1::soap_serialize_SOAP_ENV__Reason(soap, p), 0) || deepf1::soap_put_SOAP_ENV__Reason(soap, p, "SOAP-ENV:Reason", "") || soap_end_send(soap))
		return soap_closesock(soap);
	return SOAP_OK;
}
SOAP_FMAC3 struct SOAP_ENV__Reason * SOAP_FMAC4 soap_get_SOAP_ENV__Reason(struct soap*, struct SOAP_ENV__Reason *, const char*, const char*);

inline int soap_read_SOAP_ENV__Reason(struct soap *soap, struct SOAP_ENV__Reason *p)
{
	if (p)
	{	deepf1::soap_default_SOAP_ENV__Reason(soap, p);
		if (soap_begin_recv(soap) || deepf1::soap_get_SOAP_ENV__Reason(soap, p, NULL, NULL) == NULL || soap_end_recv(soap))
			return soap->error;
	}
	return SOAP_OK;
}

inline int soap_GET_SOAP_ENV__Reason(struct soap *soap, const char *URL, struct SOAP_ENV__Reason *p)
{
	if (soap_GET(soap, URL, NULL) || deepf1::soap_read_SOAP_ENV__Reason(soap, p) || soap_closesock(soap))
		return soap_closesock(soap);
	return SOAP_OK;
}

inline int soap_POST_recv_SOAP_ENV__Reason(struct soap *soap, struct SOAP_ENV__Reason *p)
{
	if (deepf1::soap_read_SOAP_ENV__Reason(soap, p) || soap_closesock(soap))
		return soap_closesock(soap);
	return SOAP_OK;
}
#endif

#endif

#ifndef WITH_NOGLOBAL

#ifndef SOAP_TYPE_deepf1_SOAP_ENV__Detail_DEFINED
#define SOAP_TYPE_deepf1_SOAP_ENV__Detail_DEFINED
SOAP_FMAC3 void SOAP_FMAC4 soap_default_SOAP_ENV__Detail(struct soap*, struct SOAP_ENV__Detail *);
SOAP_FMAC3 void SOAP_FMAC4 soap_serialize_SOAP_ENV__Detail(struct soap*, const struct SOAP_ENV__Detail *);
SOAP_FMAC3 int SOAP_FMAC4 soap_out_SOAP_ENV__Detail(struct soap*, const char*, int, const struct SOAP_ENV__Detail *, const char*);
SOAP_FMAC3 struct SOAP_ENV__Detail * SOAP_FMAC4 soap_in_SOAP_ENV__Detail(struct soap*, const char*, struct SOAP_ENV__Detail *, const char*);
SOAP_FMAC1 struct SOAP_ENV__Detail * SOAP_FMAC2 soap_instantiate_SOAP_ENV__Detail(struct soap*, int, const char*, const char*, size_t*);

inline struct SOAP_ENV__Detail * soap_new_SOAP_ENV__Detail(struct soap *soap, int n = -1)
{
	return soap_instantiate_SOAP_ENV__Detail(soap, n, NULL, NULL, NULL);
}

inline struct SOAP_ENV__Detail * soap_new_req_SOAP_ENV__Detail(
	struct soap *soap,
	int __type,
	void *fault)
{
	struct SOAP_ENV__Detail *_p = deepf1::soap_new_SOAP_ENV__Detail(soap);
	if (_p)
	{	deepf1::soap_default_SOAP_ENV__Detail(soap, _p);
		_p->__type = __type;
		_p->fault = fault;
	}
	return _p;
}

inline struct SOAP_ENV__Detail * soap_new_set_SOAP_ENV__Detail(
	struct soap *soap,
	char *__any,
	int __type,
	void *fault)
{
	struct SOAP_ENV__Detail *_p = deepf1::soap_new_SOAP_ENV__Detail(soap);
	if (_p)
	{	deepf1::soap_default_SOAP_ENV__Detail(soap, _p);
		_p->__any = __any;
		_p->__type = __type;
		_p->fault = fault;
	}
	return _p;
}
SOAP_FMAC3 int SOAP_FMAC4 soap_put_SOAP_ENV__Detail(struct soap*, const struct SOAP_ENV__Detail *, const char*, const char*);

inline int soap_write_SOAP_ENV__Detail(struct soap *soap, struct SOAP_ENV__Detail const*p)
{
	soap_free_temp(soap);
	if (soap_begin_send(soap) || (deepf1::soap_serialize_SOAP_ENV__Detail(soap, p), 0) || deepf1::soap_put_SOAP_ENV__Detail(soap, p, "SOAP-ENV:Detail", "") || soap_end_send(soap))
			return soap->error;
	return SOAP_OK;
}

inline int soap_PUT_SOAP_ENV__Detail(struct soap *soap, const char *URL, struct SOAP_ENV__Detail const*p)
{
	soap_free_temp(soap);
	if (soap_PUT(soap, URL, NULL, "text/xml; charset=utf-8") || (deepf1::soap_serialize_SOAP_ENV__Detail(soap, p), 0) || deepf1::soap_put_SOAP_ENV__Detail(soap, p, "SOAP-ENV:Detail", "") || soap_end_send(soap) || soap_recv_empty_response(soap))
		return soap_closesock(soap);
	return SOAP_OK;
}

inline int soap_POST_send_SOAP_ENV__Detail(struct soap *soap, const char *URL, struct SOAP_ENV__Detail const*p)
{
	soap_free_temp(soap);
	if (soap_POST(soap, URL, NULL, "text/xml; charset=utf-8") || (deepf1::soap_serialize_SOAP_ENV__Detail(soap, p), 0) || deepf1::soap_put_SOAP_ENV__Detail(soap, p, "SOAP-ENV:Detail", "") || soap_end_send(soap))
		return soap_closesock(soap);
	return SOAP_OK;
}
SOAP_FMAC3 struct SOAP_ENV__Detail * SOAP_FMAC4 soap_get_SOAP_ENV__Detail(struct soap*, struct SOAP_ENV__Detail *, const char*, const char*);

inline int soap_read_SOAP_ENV__Detail(struct soap *soap, struct SOAP_ENV__Detail *p)
{
	if (p)
	{	deepf1::soap_default_SOAP_ENV__Detail(soap, p);
		if (soap_begin_recv(soap) || deepf1::soap_get_SOAP_ENV__Detail(soap, p, NULL, NULL) == NULL || soap_end_recv(soap))
			return soap->error;
	}
	return SOAP_OK;
}

inline int soap_GET_SOAP_ENV__Detail(struct soap *soap, const char *URL, struct SOAP_ENV__Detail *p)
{
	if (soap_GET(soap, URL, NULL) || deepf1::soap_read_SOAP_ENV__Detail(soap, p) || soap_closesock(soap))
		return soap_closesock(soap);
	return SOAP_OK;
}

inline int soap_POST_recv_SOAP_ENV__Detail(struct soap *soap, struct SOAP_ENV__Detail *p)
{
	if (deepf1::soap_read_SOAP_ENV__Detail(soap, p) || soap_closesock(soap))
		return soap_closesock(soap);
	return SOAP_OK;
}
#endif

#endif

#ifndef WITH_NOGLOBAL

#ifndef SOAP_TYPE_deepf1_SOAP_ENV__Code_DEFINED
#define SOAP_TYPE_deepf1_SOAP_ENV__Code_DEFINED
SOAP_FMAC3 void SOAP_FMAC4 soap_default_SOAP_ENV__Code(struct soap*, struct SOAP_ENV__Code *);
SOAP_FMAC3 void SOAP_FMAC4 soap_serialize_SOAP_ENV__Code(struct soap*, const struct SOAP_ENV__Code *);
SOAP_FMAC3 int SOAP_FMAC4 soap_out_SOAP_ENV__Code(struct soap*, const char*, int, const struct SOAP_ENV__Code *, const char*);
SOAP_FMAC3 struct SOAP_ENV__Code * SOAP_FMAC4 soap_in_SOAP_ENV__Code(struct soap*, const char*, struct SOAP_ENV__Code *, const char*);
SOAP_FMAC1 struct SOAP_ENV__Code * SOAP_FMAC2 soap_instantiate_SOAP_ENV__Code(struct soap*, int, const char*, const char*, size_t*);

inline struct SOAP_ENV__Code * soap_new_SOAP_ENV__Code(struct soap *soap, int n = -1)
{
	return soap_instantiate_SOAP_ENV__Code(soap, n, NULL, NULL, NULL);
}

inline struct SOAP_ENV__Code * soap_new_req_SOAP_ENV__Code(
	struct soap *soap)
{
	struct SOAP_ENV__Code *_p = deepf1::soap_new_SOAP_ENV__Code(soap);
	if (_p)
	{	deepf1::soap_default_SOAP_ENV__Code(soap, _p);
	}
	return _p;
}

inline struct SOAP_ENV__Code * soap_new_set_SOAP_ENV__Code(
	struct soap *soap,
	char *SOAP_ENV__Value,
	struct SOAP_ENV__Code *SOAP_ENV__Subcode)
{
	struct SOAP_ENV__Code *_p = deepf1::soap_new_SOAP_ENV__Code(soap);
	if (_p)
	{	deepf1::soap_default_SOAP_ENV__Code(soap, _p);
		_p->SOAP_ENV__Value = SOAP_ENV__Value;
		_p->SOAP_ENV__Subcode = SOAP_ENV__Subcode;
	}
	return _p;
}
SOAP_FMAC3 int SOAP_FMAC4 soap_put_SOAP_ENV__Code(struct soap*, const struct SOAP_ENV__Code *, const char*, const char*);

inline int soap_write_SOAP_ENV__Code(struct soap *soap, struct SOAP_ENV__Code const*p)
{
	soap_free_temp(soap);
	if (soap_begin_send(soap) || (deepf1::soap_serialize_SOAP_ENV__Code(soap, p), 0) || deepf1::soap_put_SOAP_ENV__Code(soap, p, "SOAP-ENV:Code", "") || soap_end_send(soap))
			return soap->error;
	return SOAP_OK;
}

inline int soap_PUT_SOAP_ENV__Code(struct soap *soap, const char *URL, struct SOAP_ENV__Code const*p)
{
	soap_free_temp(soap);
	if (soap_PUT(soap, URL, NULL, "text/xml; charset=utf-8") || (deepf1::soap_serialize_SOAP_ENV__Code(soap, p), 0) || deepf1::soap_put_SOAP_ENV__Code(soap, p, "SOAP-ENV:Code", "") || soap_end_send(soap) || soap_recv_empty_response(soap))
		return soap_closesock(soap);
	return SOAP_OK;
}

inline int soap_POST_send_SOAP_ENV__Code(struct soap *soap, const char *URL, struct SOAP_ENV__Code const*p)
{
	soap_free_temp(soap);
	if (soap_POST(soap, URL, NULL, "text/xml; charset=utf-8") || (deepf1::soap_serialize_SOAP_ENV__Code(soap, p), 0) || deepf1::soap_put_SOAP_ENV__Code(soap, p, "SOAP-ENV:Code", "") || soap_end_send(soap))
		return soap_closesock(soap);
	return SOAP_OK;
}
SOAP_FMAC3 struct SOAP_ENV__Code * SOAP_FMAC4 soap_get_SOAP_ENV__Code(struct soap*, struct SOAP_ENV__Code *, const char*, const char*);

inline int soap_read_SOAP_ENV__Code(struct soap *soap, struct SOAP_ENV__Code *p)
{
	if (p)
	{	deepf1::soap_default_SOAP_ENV__Code(soap, p);
		if (soap_begin_recv(soap) || deepf1::soap_get_SOAP_ENV__Code(soap, p, NULL, NULL) == NULL || soap_end_recv(soap))
			return soap->error;
	}
	return SOAP_OK;
}

inline int soap_GET_SOAP_ENV__Code(struct soap *soap, const char *URL, struct SOAP_ENV__Code *p)
{
	if (soap_GET(soap, URL, NULL) || deepf1::soap_read_SOAP_ENV__Code(soap, p) || soap_closesock(soap))
		return soap_closesock(soap);
	return SOAP_OK;
}

inline int soap_POST_recv_SOAP_ENV__Code(struct soap *soap, struct SOAP_ENV__Code *p)
{
	if (deepf1::soap_read_SOAP_ENV__Code(soap, p) || soap_closesock(soap))
		return soap_closesock(soap);
	return SOAP_OK;
}
#endif

#endif

#ifndef WITH_NOGLOBAL

#ifndef SOAP_TYPE_deepf1_SOAP_ENV__Header_DEFINED
#define SOAP_TYPE_deepf1_SOAP_ENV__Header_DEFINED
SOAP_FMAC3 void SOAP_FMAC4 soap_default_SOAP_ENV__Header(struct soap*, struct SOAP_ENV__Header *);
SOAP_FMAC3 void SOAP_FMAC4 soap_serialize_SOAP_ENV__Header(struct soap*, const struct SOAP_ENV__Header *);
SOAP_FMAC3 int SOAP_FMAC4 soap_out_SOAP_ENV__Header(struct soap*, const char*, int, const struct SOAP_ENV__Header *, const char*);
SOAP_FMAC3 struct SOAP_ENV__Header * SOAP_FMAC4 soap_in_SOAP_ENV__Header(struct soap*, const char*, struct SOAP_ENV__Header *, const char*);
SOAP_FMAC1 struct SOAP_ENV__Header * SOAP_FMAC2 soap_instantiate_SOAP_ENV__Header(struct soap*, int, const char*, const char*, size_t*);

inline struct SOAP_ENV__Header * soap_new_SOAP_ENV__Header(struct soap *soap, int n = -1)
{
	return soap_instantiate_SOAP_ENV__Header(soap, n, NULL, NULL, NULL);
}

inline struct SOAP_ENV__Header * soap_new_req_SOAP_ENV__Header(
	struct soap *soap)
{
	struct SOAP_ENV__Header *_p = deepf1::soap_new_SOAP_ENV__Header(soap);
	if (_p)
	{	deepf1::soap_default_SOAP_ENV__Header(soap, _p);
	}
	return _p;
}

inline struct SOAP_ENV__Header * soap_new_set_SOAP_ENV__Header(
	struct soap *soap)
{
	struct SOAP_ENV__Header *_p = deepf1::soap_new_SOAP_ENV__Header(soap);
	if (_p)
	{	deepf1::soap_default_SOAP_ENV__Header(soap, _p);
	}
	return _p;
}
SOAP_FMAC3 int SOAP_FMAC4 soap_put_SOAP_ENV__Header(struct soap*, const struct SOAP_ENV__Header *, const char*, const char*);

inline int soap_write_SOAP_ENV__Header(struct soap *soap, struct SOAP_ENV__Header const*p)
{
	soap_free_temp(soap);
	if (soap_begin_send(soap) || (deepf1::soap_serialize_SOAP_ENV__Header(soap, p), 0) || deepf1::soap_put_SOAP_ENV__Header(soap, p, "SOAP-ENV:Header", "") || soap_end_send(soap))
			return soap->error;
	return SOAP_OK;
}

inline int soap_PUT_SOAP_ENV__Header(struct soap *soap, const char *URL, struct SOAP_ENV__Header const*p)
{
	soap_free_temp(soap);
	if (soap_PUT(soap, URL, NULL, "text/xml; charset=utf-8") || (deepf1::soap_serialize_SOAP_ENV__Header(soap, p), 0) || deepf1::soap_put_SOAP_ENV__Header(soap, p, "SOAP-ENV:Header", "") || soap_end_send(soap) || soap_recv_empty_response(soap))
		return soap_closesock(soap);
	return SOAP_OK;
}

inline int soap_POST_send_SOAP_ENV__Header(struct soap *soap, const char *URL, struct SOAP_ENV__Header const*p)
{
	soap_free_temp(soap);
	if (soap_POST(soap, URL, NULL, "text/xml; charset=utf-8") || (deepf1::soap_serialize_SOAP_ENV__Header(soap, p), 0) || deepf1::soap_put_SOAP_ENV__Header(soap, p, "SOAP-ENV:Header", "") || soap_end_send(soap))
		return soap_closesock(soap);
	return SOAP_OK;
}
SOAP_FMAC3 struct SOAP_ENV__Header * SOAP_FMAC4 soap_get_SOAP_ENV__Header(struct soap*, struct SOAP_ENV__Header *, const char*, const char*);

inline int soap_read_SOAP_ENV__Header(struct soap *soap, struct SOAP_ENV__Header *p)
{
	if (p)
	{	deepf1::soap_default_SOAP_ENV__Header(soap, p);
		if (soap_begin_recv(soap) || deepf1::soap_get_SOAP_ENV__Header(soap, p, NULL, NULL) == NULL || soap_end_recv(soap))
			return soap->error;
	}
	return SOAP_OK;
}

inline int soap_GET_SOAP_ENV__Header(struct soap *soap, const char *URL, struct SOAP_ENV__Header *p)
{
	if (soap_GET(soap, URL, NULL) || deepf1::soap_read_SOAP_ENV__Header(soap, p) || soap_closesock(soap))
		return soap_closesock(soap);
	return SOAP_OK;
}

inline int soap_POST_recv_SOAP_ENV__Header(struct soap *soap, struct SOAP_ENV__Header *p)
{
	if (deepf1::soap_read_SOAP_ENV__Header(soap, p) || soap_closesock(soap))
		return soap_closesock(soap);
	return SOAP_OK;
}
#endif

#endif

#ifndef SOAP_TYPE_deepf1_cpu_times_DEFINED
#define SOAP_TYPE_deepf1_cpu_times_DEFINED
SOAP_FMAC3 void SOAP_FMAC4 soap_default_cpu_times(struct soap*, struct cpu_times *);
SOAP_FMAC3 void SOAP_FMAC4 soap_serialize_cpu_times(struct soap*, const struct cpu_times *);
SOAP_FMAC3 int SOAP_FMAC4 soap_out_cpu_times(struct soap*, const char*, int, const struct cpu_times *, const char*);
SOAP_FMAC3 struct cpu_times * SOAP_FMAC4 soap_in_cpu_times(struct soap*, const char*, struct cpu_times *, const char*);
SOAP_FMAC1 struct cpu_times * SOAP_FMAC2 soap_instantiate_cpu_times(struct soap*, int, const char*, const char*, size_t*);

inline struct cpu_times * soap_new_cpu_times(struct soap *soap, int n = -1)
{
	return soap_instantiate_cpu_times(soap, n, NULL, NULL, NULL);
}

inline struct cpu_times * soap_new_req_cpu_times(
	struct soap *soap,
	long wall,
	long user,
	long system)
{
	struct cpu_times *_p = deepf1::soap_new_cpu_times(soap);
	if (_p)
	{	deepf1::soap_default_cpu_times(soap, _p);
		_p->wall = wall;
		_p->user = user;
		_p->system = system;
	}
	return _p;
}

inline struct cpu_times * soap_new_set_cpu_times(
	struct soap *soap,
	long wall,
	long user,
	long system)
{
	struct cpu_times *_p = deepf1::soap_new_cpu_times(soap);
	if (_p)
	{	deepf1::soap_default_cpu_times(soap, _p);
		_p->wall = wall;
		_p->user = user;
		_p->system = system;
	}
	return _p;
}
SOAP_FMAC3 int SOAP_FMAC4 soap_put_cpu_times(struct soap*, const struct cpu_times *, const char*, const char*);

inline int soap_write_cpu_times(struct soap *soap, struct cpu_times const*p)
{
	soap_free_temp(soap);
	if (soap_begin_send(soap) || (deepf1::soap_serialize_cpu_times(soap, p), 0) || deepf1::soap_put_cpu_times(soap, p, "cpu-times", "") || soap_end_send(soap))
			return soap->error;
	return SOAP_OK;
}

inline int soap_PUT_cpu_times(struct soap *soap, const char *URL, struct cpu_times const*p)
{
	soap_free_temp(soap);
	if (soap_PUT(soap, URL, NULL, "text/xml; charset=utf-8") || (deepf1::soap_serialize_cpu_times(soap, p), 0) || deepf1::soap_put_cpu_times(soap, p, "cpu-times", "") || soap_end_send(soap) || soap_recv_empty_response(soap))
		return soap_closesock(soap);
	return SOAP_OK;
}

inline int soap_POST_send_cpu_times(struct soap *soap, const char *URL, struct cpu_times const*p)
{
	soap_free_temp(soap);
	if (soap_POST(soap, URL, NULL, "text/xml; charset=utf-8") || (deepf1::soap_serialize_cpu_times(soap, p), 0) || deepf1::soap_put_cpu_times(soap, p, "cpu-times", "") || soap_end_send(soap))
		return soap_closesock(soap);
	return SOAP_OK;
}
SOAP_FMAC3 struct cpu_times * SOAP_FMAC4 soap_get_cpu_times(struct soap*, struct cpu_times *, const char*, const char*);

inline int soap_read_cpu_times(struct soap *soap, struct cpu_times *p)
{
	if (p)
	{	deepf1::soap_default_cpu_times(soap, p);
		if (soap_begin_recv(soap) || deepf1::soap_get_cpu_times(soap, p, NULL, NULL) == NULL || soap_end_recv(soap))
			return soap->error;
	}
	return SOAP_OK;
}

inline int soap_GET_cpu_times(struct soap *soap, const char *URL, struct cpu_times *p)
{
	if (soap_GET(soap, URL, NULL) || deepf1::soap_read_cpu_times(soap, p) || soap_closesock(soap))
		return soap_closesock(soap);
	return SOAP_OK;
}

inline int soap_POST_recv_cpu_times(struct soap *soap, struct cpu_times *p)
{
	if (deepf1::soap_read_cpu_times(soap, p) || soap_closesock(soap))
		return soap_closesock(soap);
	return SOAP_OK;
}
#endif

#ifndef WITH_NOGLOBAL

#ifndef SOAP_TYPE_deepf1_PointerToSOAP_ENV__Reason_DEFINED
#define SOAP_TYPE_deepf1_PointerToSOAP_ENV__Reason_DEFINED
SOAP_FMAC3 void SOAP_FMAC4 soap_serialize_PointerToSOAP_ENV__Reason(struct soap*, struct SOAP_ENV__Reason *const*);
SOAP_FMAC3 int SOAP_FMAC4 soap_out_PointerToSOAP_ENV__Reason(struct soap*, const char *, int, struct SOAP_ENV__Reason *const*, const char *);
SOAP_FMAC3 struct SOAP_ENV__Reason ** SOAP_FMAC4 soap_in_PointerToSOAP_ENV__Reason(struct soap*, const char*, struct SOAP_ENV__Reason **, const char*);
SOAP_FMAC3 int SOAP_FMAC4 soap_put_PointerToSOAP_ENV__Reason(struct soap*, struct SOAP_ENV__Reason *const*, const char*, const char*);
SOAP_FMAC3 struct SOAP_ENV__Reason ** SOAP_FMAC4 soap_get_PointerToSOAP_ENV__Reason(struct soap*, struct SOAP_ENV__Reason **, const char*, const char*);
#endif

#endif

#ifndef WITH_NOGLOBAL

#ifndef SOAP_TYPE_deepf1_PointerToSOAP_ENV__Detail_DEFINED
#define SOAP_TYPE_deepf1_PointerToSOAP_ENV__Detail_DEFINED
SOAP_FMAC3 void SOAP_FMAC4 soap_serialize_PointerToSOAP_ENV__Detail(struct soap*, struct SOAP_ENV__Detail *const*);
SOAP_FMAC3 int SOAP_FMAC4 soap_out_PointerToSOAP_ENV__Detail(struct soap*, const char *, int, struct SOAP_ENV__Detail *const*, const char *);
SOAP_FMAC3 struct SOAP_ENV__Detail ** SOAP_FMAC4 soap_in_PointerToSOAP_ENV__Detail(struct soap*, const char*, struct SOAP_ENV__Detail **, const char*);
SOAP_FMAC3 int SOAP_FMAC4 soap_put_PointerToSOAP_ENV__Detail(struct soap*, struct SOAP_ENV__Detail *const*, const char*, const char*);
SOAP_FMAC3 struct SOAP_ENV__Detail ** SOAP_FMAC4 soap_get_PointerToSOAP_ENV__Detail(struct soap*, struct SOAP_ENV__Detail **, const char*, const char*);
#endif

#endif

#ifndef WITH_NOGLOBAL

#ifndef SOAP_TYPE_deepf1_PointerToSOAP_ENV__Code_DEFINED
#define SOAP_TYPE_deepf1_PointerToSOAP_ENV__Code_DEFINED
SOAP_FMAC3 void SOAP_FMAC4 soap_serialize_PointerToSOAP_ENV__Code(struct soap*, struct SOAP_ENV__Code *const*);
SOAP_FMAC3 int SOAP_FMAC4 soap_out_PointerToSOAP_ENV__Code(struct soap*, const char *, int, struct SOAP_ENV__Code *const*, const char *);
SOAP_FMAC3 struct SOAP_ENV__Code ** SOAP_FMAC4 soap_in_PointerToSOAP_ENV__Code(struct soap*, const char*, struct SOAP_ENV__Code **, const char*);
SOAP_FMAC3 int SOAP_FMAC4 soap_put_PointerToSOAP_ENV__Code(struct soap*, struct SOAP_ENV__Code *const*, const char*, const char*);
SOAP_FMAC3 struct SOAP_ENV__Code ** SOAP_FMAC4 soap_get_PointerToSOAP_ENV__Code(struct soap*, struct SOAP_ENV__Code **, const char*, const char*);
#endif

#endif

#ifndef SOAP_TYPE_deepf1__QName_DEFINED
#define SOAP_TYPE_deepf1__QName_DEFINED

inline void soap_default__QName(struct soap *soap, char **a)
{
	(void)soap; /* appease -Wall -Werror */
#ifdef SOAP_DEFAULT__QName
	*a = SOAP_DEFAULT__QName;
#else
	*a = (char *)0;
#endif
}
SOAP_FMAC3 void SOAP_FMAC4 soap_serialize__QName(struct soap*, char *const*);

#define soap__QName2s(soap, a) soap_QName2s(soap, (a))
SOAP_FMAC3 int SOAP_FMAC4 soap_out__QName(struct soap*, const char*, int, char*const*, const char*);

#define soap_s2_QName(soap, s, a) soap_s2QName((soap), (s), (char**)(a), 0, -1, NULL)
SOAP_FMAC3 char * * SOAP_FMAC4 soap_in__QName(struct soap*, const char*, char **, const char*);

#define soap_instantiate__QName soap_instantiate_string


#define soap_new__QName soap_new_string

SOAP_FMAC3 int SOAP_FMAC4 soap_put__QName(struct soap*, char *const*, const char*, const char*);

inline int soap_write__QName(struct soap *soap, char *const*p)
{
	soap_free_temp(soap);
	if (p)
	{	if (soap_begin_send(soap) || deepf1::soap_put__QName(soap, p, "QName", "") || soap_end_send(soap))
			return soap->error;
	}
	return SOAP_OK;
}

inline int soap_PUT__QName(struct soap *soap, const char *URL, char *const*p)
{
	soap_free_temp(soap);
	if (soap_PUT(soap, URL, NULL, "text/xml; charset=utf-8") || deepf1::soap_put__QName(soap, p, "QName", "") || soap_end_send(soap) || soap_recv_empty_response(soap))
		return soap_closesock(soap);
	return SOAP_OK;
}

inline int soap_POST_send__QName(struct soap *soap, const char *URL, char *const*p)
{
	soap_free_temp(soap);
	if (soap_POST(soap, URL, NULL, "text/xml; charset=utf-8") || deepf1::soap_put__QName(soap, p, "QName", "") || soap_end_send(soap))
		return soap_closesock(soap);
	return SOAP_OK;
}
SOAP_FMAC3 char ** SOAP_FMAC4 soap_get__QName(struct soap*, char **, const char*, const char*);

inline int soap_read__QName(struct soap *soap, char **p)
{
	if (p)
	{	if (soap_begin_recv(soap) || deepf1::soap_get__QName(soap, p, NULL, NULL) == NULL || soap_end_recv(soap))
			return soap->error;
	}
	return SOAP_OK;
}

inline int soap_GET__QName(struct soap *soap, const char *URL, char **p)
{
	if (soap_GET(soap, URL, NULL) || deepf1::soap_read__QName(soap, p) || soap_closesock(soap))
		return soap_closesock(soap);
	return SOAP_OK;
}

inline int soap_POST_recv__QName(struct soap *soap, char **p)
{
	if (deepf1::soap_read__QName(soap, p) || soap_closesock(soap))
		return soap_closesock(soap);
	return SOAP_OK;
}
#endif

#ifndef SOAP_TYPE_deepf1__XML_DEFINED
#define SOAP_TYPE_deepf1__XML_DEFINED
#endif

#ifndef SOAP_TYPE_deepf1_string_DEFINED
#define SOAP_TYPE_deepf1_string_DEFINED

inline void soap_default_string(struct soap *soap, char **a)
{
	(void)soap; /* appease -Wall -Werror */
#ifdef SOAP_DEFAULT_string
	*a = SOAP_DEFAULT_string;
#else
	*a = (char *)0;
#endif
}
SOAP_FMAC3 void SOAP_FMAC4 soap_serialize_string(struct soap*, char *const*);

#define soap_string2s(soap, a) (a)
SOAP_FMAC3 int SOAP_FMAC4 soap_out_string(struct soap*, const char*, int, char*const*, const char*);

#define soap_s2string(soap, s, a) soap_s2char((soap), (s), (char**)(a), 1, 0, -1, NULL)
SOAP_FMAC3 char * * SOAP_FMAC4 soap_in_string(struct soap*, const char*, char **, const char*);

SOAP_FMAC3 char * * SOAP_FMAC4 soap_new_string(struct soap *soap, int n = -1);
SOAP_FMAC3 int SOAP_FMAC4 soap_put_string(struct soap*, char *const*, const char*, const char*);

inline int soap_write_string(struct soap *soap, char *const*p)
{
	soap_free_temp(soap);
	if (p)
	{	if (soap_begin_send(soap) || deepf1::soap_put_string(soap, p, "string", "") || soap_end_send(soap))
			return soap->error;
	}
	return SOAP_OK;
}

inline int soap_PUT_string(struct soap *soap, const char *URL, char *const*p)
{
	soap_free_temp(soap);
	if (soap_PUT(soap, URL, NULL, "text/xml; charset=utf-8") || deepf1::soap_put_string(soap, p, "string", "") || soap_end_send(soap) || soap_recv_empty_response(soap))
		return soap_closesock(soap);
	return SOAP_OK;
}

inline int soap_POST_send_string(struct soap *soap, const char *URL, char *const*p)
{
	soap_free_temp(soap);
	if (soap_POST(soap, URL, NULL, "text/xml; charset=utf-8") || deepf1::soap_put_string(soap, p, "string", "") || soap_end_send(soap))
		return soap_closesock(soap);
	return SOAP_OK;
}
SOAP_FMAC3 char ** SOAP_FMAC4 soap_get_string(struct soap*, char **, const char*, const char*);

inline int soap_read_string(struct soap *soap, char **p)
{
	if (p)
	{	if (soap_begin_recv(soap) || deepf1::soap_get_string(soap, p, NULL, NULL) == NULL || soap_end_recv(soap))
			return soap->error;
	}
	return SOAP_OK;
}

inline int soap_GET_string(struct soap *soap, const char *URL, char **p)
{
	if (soap_GET(soap, URL, NULL) || deepf1::soap_read_string(soap, p) || soap_closesock(soap))
		return soap_closesock(soap);
	return SOAP_OK;
}

inline int soap_POST_recv_string(struct soap *soap, char **p)
{
	if (deepf1::soap_read_string(soap, p) || soap_closesock(soap))
		return soap_closesock(soap);
	return SOAP_OK;
}
#endif

} // namespace deepf1


#endif

/* End of deepf1H.h */
