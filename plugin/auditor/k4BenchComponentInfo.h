// k4BenchComponentInfo.h
//
// What the k4BenchAuditor finds out about a component beyond what Gaudi's
// managers say: the shared object defining its class, and the class doing its
// work when it wraps another, such as a Marlin processor. Marlin is reached
// through its exported symbols, so the auditor does not depend on it.

#pragma once

#include <GaudiKernel/IInterface.h>
#include <GaudiKernel/IProperty.h>

#include <dlfcn.h>
#include <link.h>

#include <cstddef>
#include <filesystem>
#include <string>
#include <system_error>

namespace k4bench
{
  inline std::string unquoted(std::string text)
  {
    if (text.size() >= 2 && (text.front() == '\'' || text.front() == '"') && text.back() == text.front())
    {
      return text.substr(1, text.size() - 2);
    }
    return text;
  }

  // Shared object holding *address*, resolved through symlinks so a path into
  // an LCG view names the package install it points to.
  inline std::string library_at(const void *address)
  {
    Dl_info info{};
    if (address == nullptr || ::dladdr(address, &info) == 0 || info.dli_fname == nullptr)
    {
      return {};
    }
    std::error_code error;
    const auto path = std::filesystem::canonical(info.dli_fname, error);
    return error ? std::string(info.dli_fname) : path.string();
  }

  // Shared object the dynamic type of *object* is defined in.
  inline std::string library_of(const void *object)
  {
    // Under the Itanium C++ ABI a polymorphic object starts with its vtable
    // pointer, and the vtable lives in the library that defines the class.
    return library_at(*static_cast<const void *const *>(object));
  }

  // Address of *symbol* in any loaded object, or null.
  inline void *loaded_symbol(const char *symbol)
  {
    if (void *address = ::dlsym(RTLD_DEFAULT, symbol))
    {
      return address;
    }
    // A library opened with RTLD_LOCAL is out of the global scope; ask each
    // loaded object in turn.
    struct Search
    {
      const char *symbol;
      void *address{nullptr};
    } search{symbol};
    ::dl_iterate_phdr(
        [](dl_phdr_info *object, std::size_t, void *data)
        {
          auto &search = *static_cast<Search *>(data);
          if (object->dlpi_name == nullptr || *object->dlpi_name == '\0')
          {
            return 0;
          }
          void *handle = ::dlopen(object->dlpi_name, RTLD_LAZY | RTLD_NOLOAD);
          if (handle == nullptr)
          {
            return 0;
          }
          search.address = ::dlsym(handle, search.symbol);
          ::dlclose(handle);
          return search.address != nullptr ? 1 : 0;
        },
        &search);
    return search.address;
  }

  // Shared object defining the Marlin processor registered as *type*: that of
  // the prototype Marlin's ProcessorMgr keeps for it. A processor's type is the
  // name it registers under, not its class ("AIDAProcessor" is
  // marlin::AIDAProcessor), and only the prototype knows the class. Marlin is
  // reached through its exported symbols, so the auditor does not depend on it;
  // the member function is called as a function taking the object first, as the
  // Itanium C++ ABI passes it. Empty when the job loaded no Marlin or no
  // processor registered as *type*.
  inline std::string library_of_marlin_processor(const std::string &type)
  {
    using Instance = void *(*)();
    using GetProcessor = void *(*)(void *, const std::string &);
    const auto instance = reinterpret_cast<Instance>(loaded_symbol("_ZN6marlin12ProcessorMgr8instanceEv"));
    const auto get_processor = reinterpret_cast<GetProcessor>(loaded_symbol(
        "_ZN6marlin12ProcessorMgr12getProcessorERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE"));
    if (instance == nullptr || get_processor == nullptr)
    {
      return {};
    }
    const void *prototype = get_processor(instance(), type);
    return prototype != nullptr ? library_of(prototype) : std::string{};
  }

  // The class doing the work: the component's own type, unless a wrapper names
  // the class it delegates to in a ProcessorType property (k4MarlinWrapper).
  inline std::string implementation(IInterface *component, const std::string &type)
  {
    auto *properties = dynamic_cast<IProperty *>(component);
    std::string wrapped;
    if (properties && properties->hasProperty("ProcessorType") &&
        properties->getProperty("ProcessorType", wrapped).isSuccess() && !wrapped.empty())
    {
      return unquoted(wrapped);
    }
    return type;
  }
} // namespace k4bench
