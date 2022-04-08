using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Text;
using System.Threading.Tasks;
using System.Linq;

namespace dns_netcore
{

	class RecursiveResolver : IRecursiveResolver
	{
		private IDNSClient dnsClient;
		private ConcurrentDictionary<string, IP4Addr> cache;

		public RecursiveResolver(IDNSClient client)
		{
			this.dnsClient = client;
			this.cache = new ConcurrentDictionary<string, IP4Addr>();
		}

		public Task<IP4Addr> ResolveRecursive(string domain)
		{
            return Task<IP4Addr>.Run(() => {
            string[] domains = domain.Split('.');
				Array.Reverse(domains);
				IP4Addr res = this.dnsClient.GetRootServers()[0];

				String subdomain = null;

				for (var i = 0; i < domains.Length; i++) {
					subdomain = domains[i];
					// The full path of the subdomain we are resolving (from backwards)
					// i.e. mff.cuni.cz = cz -> cuni.cz -> mff.cuni.cz
					var subdomainPath = String.Join(".", domains.Take(i+1).Reverse().ToList());

					// Resolves the domain directly 
					var fallbackQuery = this.dnsClient.Resolve(res, subdomain);

					bool cacheUsed = false;
					// Checks if the subdomain's IP is cached
					if (this.cache.ContainsKey(subdomainPath)) {
						IP4Addr cachedIP;

						if (this.cache.TryGetValue(subdomainPath, out cachedIP)) {
							// Possible cache hit, checks if the record is still valid
							var cacheQuery = this.dnsClient.Reverse(cachedIP);

							try
							{
								cacheQuery.Wait(); 
								if (cacheQuery.Result == subdomainPath)
								{
									res = cachedIP;
									cacheUsed = true;
								}
								else
								{
									this.cache.TryRemove(subdomainPath, out _);
								}
							}
							catch (AggregateException)
							{
								this.cache.TryRemove(subdomainPath, out _);
							}
						}
					}

					// When cache fails, it resolves directly
					if (!cacheUsed) {
						fallbackQuery.Wait();
						res = fallbackQuery.Result;
						this.cache.TryAdd(subdomainPath, res);
					}
				}
				this.cache.TryAdd(domain, res);

				return res;
			});
		}
	}
}